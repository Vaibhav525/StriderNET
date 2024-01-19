import torch
import torch_geometric
from typing import Any, NamedTuple,  Optional, Union
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from functools import partial
import numpy as np
import Systems

class MDTuple(NamedTuple):
    N          : torch.Tensor    #No. of particles (B_sz,)
    N_types    : torch.Tensor    #No. of particles types (B_sz,)
    box_size   : torch.Tensor    #Size of box [(B_sz,spatial_dim,2) array of [[xlo, xhi],[ylo,yhi],...,[zlo,zhi]]
    pe         : torch.Tensor    #potential energy of system (B_sz,)
    species    : torch.Tensor    #Atom types list (B_sz,N) from the set of types={0,1,2,...,N_types-1}
    R          : torch.Tensor    #Position vectors of particles (B_sz,N, spatial_dim)
    neigh_list : torch.Tensor    #neigh adjacency matrix  (B_sz,N,N)
    Graph      : Optional[Data]   #Graph structure


class Generic_system:
    def create_batched_States(self,System='LJ',Batch_size: int=10,N_sample :int=200,Traj_path=None,Species_path="species.npy"):
        """
        Available Systems = ['LJ', 'CSH', 'SW_Si']
        -Creates Train, Val, and Test data of batched system_states along with batched graphs and random shuffling
        -Current split is 60:20:20
        """
        Available_Systems = ['LJ']#, 'CSH', 'SW_Si']
        System_fns={'LJ':Systems.lennard_jones}#, 'CSH':Systems.CSH, 'SW_Si':Systems.SW_Silicon}
        #System_Liq_Temps={'LJ':2.0, 'CSH':1000, 'SW_Si':2000}
        if(System not in Available_Systems):
            raise  ValueError("Available systems are only "+str(Available_Systems))
        Chosen_sys_fn=System_fns[System]
        
        #1 Create sysytem environment:
        
        if(System=='LJ'):
            self.species=torch.from_numpy(np.load(Species_path))
            cutoffs_G:torch.Tensor =torch.Tensor([[1.5 ,1.25],[1.25 ,2.0]])
            N=self.species.shape[0]
            box_size=(N/1.2)**(1/3)
            self.pair_cutoffs_G=cutoffs_G[torch.meshgrid(self.species,self.species,indexing='xy')]
            Traj=torch.from_numpy(np.load(Traj_path))[:N_sample,:,:]
        Traj=Traj.to(torch.float32)   #numpy data has higher precision but torch default is float32   
        self.Disp_Vec_fn,self.pair_dist_fn,self.Node_energy_fn,self.Total_energy_fn,self.displacement_fn, self.shift_fn ,self.pair_cutoffs,self.pair_sigma,self.pair_epsilon=Chosen_sys_fn(species=self.species,box_size=box_size)
        
        self.Batch_Total_energy_fn=torch.vmap(self.Total_energy_fn,0)
        self.Batch_Disp_Vec_fn=torch.vmap(self.Disp_Vec_fn,(0,0))
        print(self.Batch_Total_energy_fn(Traj))
        
        #5:Creating dataset from Trajectory
        N_sample=Traj.shape[0]
        Shuffled_index=torch.randperm(N_sample)  
        data_list = [self.create_G(Traj[i],self.species,self.pair_cutoffs_G,self.pair_sigma,self.Disp_Vec_fn,self.Node_energy_fn,self.Total_energy_fn)[0] for i in Shuffled_index]
        Train_loader = DataLoader(data_list[:int(0.6*N_sample)], batch_size=Batch_size,shuffle=True,num_workers=7,persistent_workers=True)
        Test_loader = DataLoader(data_list[int(0.6*N_sample):int(0.8*N_sample)], batch_size=Batch_size,shuffle=False,num_workers=7,persistent_workers=True)
        Val_loader = DataLoader(data_list[int(0.8*N_sample):], batch_size=Batch_size,shuffle=False,num_workers=7,persistent_workers=True)
        return Train_loader,Test_loader,Val_loader
    
    def update_G(self,G_batch,Disp_vecs):
        """
        G_batch : Batched graphs
        Disp_vecs: Stacked Disp vectors (N_nodes*B_sz,spatial_dim)
        """
        B_sz=G_batch.num_graphs
        N=int(G_batch['x'].shape[0]/B_sz)
        New_Batch_R=self.shift_fn(G_batch['pos'],Disp_vecs)
        data_list = [self.create_G(New_Batch_R[N*i:N*(i+1)],self.species,self.pair_cutoffs_G,self.pair_sigma,self.Disp_Vec_fn,self.Node_energy_fn,self.Total_energy_fn)[0] for i in range(B_sz)]
        new_batch=next(iter(DataLoader(data_list, batch_size=10,shuffle=False)))
        return new_batch

    def create_G(self,R,species,pair_cutoffs_G,pair_sigma,Disp_Vec_fn,Node_energy_fn,Total_energy_fn):
        """
        R: Node Positions
        Disp_vec_fn: Calculates distance between atoms considering periodic boundaries
        species: Node type info 0 and 1
        cutoffs: pair cutoffs (N,N) shape
        sigma  : pair sigma   (N,N)
            """
        #1: Calculate pair distances
        
        dR_pair = Disp_Vec_fn(R, R)
        dr_pair =torch.sqrt(torch.sum(dR_pair ** 2, axis=-1)+1e-10)
        
        #3: Creating neigh_list and senders and receivers
        pair_cutoffs_G=pair_cutoffs_G.to(R)
        n_list=(dr_pair<pair_cutoffs_G).int()
        n_list.fill_diagonal_(0)
        (senders,receivers)=torch.where(n_list==1)

        #5: Node features
        N=R.shape[0]
        node_pe=Node_energy_fn(R)
        #broadcasting node_pe to N dim
        pair_node_pe=node_pe.repeat(N).reshape(N,N)
        neigh_pe_dist=pair_node_pe*n_list
        deg_node=torch.sum(n_list,dim=1).reshape(-1,1)
        neigh_pe_sum=torch.sum(neigh_pe_dist,dim=1).reshape((-1,1))/N
        neigh_pe_mean=neigh_pe_sum/deg_node
        species=species.to(R)
        Node_feats=torch.cat([species.reshape((-1,1)),node_pe.reshape((-1,1)),neigh_pe_sum,neigh_pe_mean],dim=1)
        #Node_feats=torch.cat([node_pe.reshape((-1,1)),neigh_pe_sum,neigh_pe_mean],dim=1)

        #6: Edge Features
        dist_vec=dR_pair[senders,receivers,:]
        pair_sigma=pair_sigma.to(R)
        l_sigma=(dr_pair-pair_sigma)[senders,receivers].reshape((-1,1))
        Edge_feats=torch.cat([dist_vec,l_sigma],dim=1)
        Energy=torch.sum(node_pe)/2
        G=Data(x=Node_feats,edge_index=torch.stack([senders,receivers]),edge_attr=Edge_feats,y=Energy,pos=R)
        return G, Energy, n_list
    
    def create_data_batch(self,Positions_lists):
        data_list = [self.create_G(Positions_lists[i],self.species,self.pair_cutoffs_G,self.pair_sigma,self.Disp_Vec_fn,self.Node_energy_fn,self.Total_energy_fn)[0] for i in range(len(Positions_lists))]
        Data_batch = DataLoader(data_list, batch_size=len(data_list))
        return next(iter(Data_batch))
    
    