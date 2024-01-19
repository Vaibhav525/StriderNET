from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch_geometric
from torch_geometric.data import Data
from typing import Callable, Optional, Union, List, Dict, Any
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.nn.models import MetaLayer
import torch.nn as nn
from Utils import *
import Utils
from Optimizers import *
import pytorch_lightning as pl

class EdgeModel(nn.Module):
    
    def __init__(self, node_emb_size: int, edge_emb_size: int, fe_layers: int,fe_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
                ):
        super().__init__()
        fe_features  =[edge_emb_size+2*node_emb_size]+[edge_emb_size]*fe_layers
        fe_mlp=[]
        
        for i in range(len(fe_features)-2):
            fe_mlp+=[nn.Linear(in_features=fe_features[i],out_features=fe_features[i+1]),fe_activation]
        fe_mlp+=[nn.Linear(in_features=fe_features[-2],out_features=fe_features[-1])]

        self.fe_mlp=nn.Sequential(*fe_mlp)
        
        
    def forward(self, edge_attr,src, dest, u=None, batch=None):
        out = torch.cat([edge_attr,src, dest], dim=1)
        out = self.fe_mlp(out)
        #if self.residuals:
        #    out = out + edge_attr
        return out


class NodeModel(nn.Module):
    def __init__(self, node_emb_size: int, edge_emb_size: int, fv1_layers: int,fv2_layers: int,fv_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False) 
                ):
        super(NodeModel, self).__init__()
        
        fv1_features  =[node_emb_size+2*edge_emb_size]+[node_emb_size]*fv1_layers
        #fv2_features  =[node_emb_size+edge_emb_size]+[node_emb_size]*fv2_layers
        
        fv1_mlp=[]
        for i in range(len(fv1_features)-2):
            fv1_mlp+=[nn.Linear(in_features=fv1_features[i],out_features=fv1_features[i+1]),fv_activation]
        fv1_mlp+=[nn.Linear(in_features=fv1_features[-2],out_features=fv1_features[-1])]

        # fv2_mlp=[]
        # for i in range(len(fv2_features)-2):
        #     fv2_mlp+=[nn.Linear(in_features=fv2_features[i],out_features=fv2_features[i+1]),fv_activation]
        # fv2_mlp+=[nn.Linear(in_features=fv2_features[-2],out_features=fv2_features[-1])]

        self.fv1_mlp=nn.Sequential(*fv1_mlp)
        # self.fv2_mlp=nn.Sequential(*fv2_mlp)
        


    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index
        #out = torch.cat([x[col], edge_attr], dim=1)
        #out = self.fv1_mlp(out)
        #out = scatter(out, row, dim=0, dim_size=x.size(0),reduce='mean')
        rec_attr=scatter(edge_attr, row, dim=0, dim_size=x.size(0),reduce='mean')
        sent_attr=scatter(edge_attr, col, dim=0, dim_size=x.size(0),reduce='mean')
        out = torch.cat([x, rec_attr,sent_attr], dim=1)
        out = self.fv1_mlp(out)
        #if self.residuals:
        #    out = out + x
        return out

class Pol_Net_lit(pl.LightningModule):
    def __init__(self,
                sys_env ,
                in_edge_feats:int,
                in_node_feats:int, 
                edge_emb_size:int,
                node_emb_size: int,
                fa_layers: int,
                fb_layers: int,
                fv1_layers: int,
                fv2_layers: int,
                fe_layers: int,
                MLP1_layers: int, #MLP1 has one additional layer for returning scaler probabilities for each node 
                MLP2_layers:int,  #MLP2 for displacement
                sigma :float,     #Scaling parameter for displacement
                multivariate_std :float,
                disp_cutoff: float,
                batchnorm_running_stats : bool,
                train_len_ep:int,
                val_len_ep:int,
                message_passing_steps: int,
                spatial_dim :int=3,                 
                fa_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
                fb_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
                #MLP1_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),#jax.nn.hard_tanh # jax.nn.softplus #relu leads to inf and nan after normalization
                #MLP1_normalize_activation:  Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softmax(dim=0),
                MLP2_activation:   Callable[[torch.Tensor],torch.Tensor] =  torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),#jax.nn.hard_tanh
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #Initialize system environment
        self.Sys=sys_env

        
        fa_features  =[in_node_feats]+[node_emb_size]*fa_layers
        fb_features  =[in_edge_feats]+[edge_emb_size]*fb_layers
        #MLP1_features=[node_emb_size]*MLP1_layers+[1]
        MLP2_features=[node_emb_size+2*edge_emb_size]*MLP2_layers+[spatial_dim] #[x,y]
        self.sigma=sigma
        self.alpha=multivariate_std
        self.disp_cutoff=disp_cutoff
        self.message_passing_steps=message_passing_steps
        self.batchnorm_running_stats=batchnorm_running_stats
        self.train_len_ep=train_len_ep
        self.val_len_ep=val_len_ep
        #fa_mlp=[nn.BatchNorm(use_running_average=not  train,
        #             momentum=0.9,epsilon=1e-5,dtype=jnp.float32)]+[nn.Linear(feat) for i,feat in enumerate(fa_features)]
        
        fa_mlp=[nn.BatchNorm1d(num_features=fa_features[0],track_running_stats=self.batchnorm_running_stats)]
        for i in range(len(fa_features)-2):
            fa_mlp+=[nn.Linear(in_features=fa_features[i],out_features=fa_features[i+1]),fa_activation]
        fa_mlp+=[nn.Linear(in_features=fa_features[-2],out_features=fa_features[-1])]

        fb_mlp=[]
        for i in range(len(fb_features)-2):
            fb_mlp+=[nn.Linear(in_features=fb_features[i],out_features=fb_features[i+1]),fb_activation]
        fb_mlp+=[nn.Linear(in_features=fb_features[-2],out_features=fb_features[-1])]

        
        
        # MLP1=[]
        # for i in range(len(MLP1_features)-2):
        #     MLP1+=[nn.Linear(in_features=MLP1_features[i],out_features=MLP1_features[i+1]),MLP1_activation]
        # MLP1+=[nn.Linear(in_features=MLP1_features[-2],out_features=MLP1_features[-1]),MLP1_normalize_activation]

        MLP2=[]
        for i in range(len(MLP2_features)-2):
            MLP2+=[nn.Linear(in_features=MLP2_features[i],out_features=MLP2_features[i+1]),MLP2_activation]
        MLP2+=[nn.Linear(in_features=MLP2_features[-2],out_features=MLP2_features[-1])]


        self.fa_mlp=nn.Sequential(*fa_mlp)
        self.fb_mlp=nn.Sequential(*fb_mlp)
        #self.MLP1=nn.Sequential(*MLP1)
        self.MLP2=nn.Sequential(*MLP2)
        self.GNNConv=MetaLayer(edge_model=EdgeModel(node_emb_size,edge_emb_size,fe_layers),
                               node_model=NodeModel(node_emb_size,edge_emb_size,fv1_layers,fv2_layers),
                               global_model=None
                                )
        
        self.save_hyperparameters()
        
    def forward(self,Inp_Graph : Data):
        Graph=Inp_Graph
        #Graph=Inp_Graph.clone() 
        #print("Graph_x",Graph['x'][:10])
        #MLP1
        # def node_to_pi_fn(nodes):
        #     #Final output is pi( scaler)
        #     x = nodes
        #     #print("ff")
        #     #print(x)
        #     return self.MLP1(nodes)
        
        #MLP2
        def Displace_node(Graph):
            row, col =Graph['edge_index']
            rec_attr=scatter(Graph['edge_attr'], row, dim=0, dim_size=Graph['x'].size(0),reduce='mean')
            sent_attr=scatter(Graph['edge_attr'], col, dim=0, dim_size=Graph['x'].size(0),reduce='mean')
            x = torch.cat([Graph['x'], rec_attr,sent_attr], dim=1)
            x=self.MLP2(x)
            r_vec=self.sigma*(x-torch.mean(x,dim=0))#/torch.max(x) 
            return r_vec 
        
        #1 Create_initial_node_emb
        Graph['x']=self.fa_mlp(Graph['x']) 

        
        #2 Create initial edge emb
        Graph['edge_attr']=self.fb_mlp(Graph['edge_attr'])

        
        #3 Message Passing
        # print(Graph['x'])
        # for k in range(Graph['x'].shape[1]):
        #     sns.kdeplot(Graph['x'].detach().cpu()[k])
        # plt.show()
        #print("Mu inp",Graph['x'])
        for k in range(self.message_passing_steps):
            Graph['x'],Graph['edge_attr'],_=self.GNNConv(Graph['x'],Graph['edge_index'],Graph['edge_attr'])
        # print(Graph['x'])
        # for k in range(Graph['x'].shape[1]):
        #     sns.kdeplot(Graph['x'].detach().cpu()[k])
        # plt.show()
        #4 Predict node probs
        
        #Debugging, turned off node_probs
        node_probs=None#Graph['x'][:,0]#node_to_pi_fn(Graph['x'])

        #5 Predict displacements
        
        #for k in range(16):

        #    sns.kdeplot(x.detach().cpu()[:,k])
        #    plt.show()            
            
        #print('x',Graph['x'])
        #print("Graph_x",Graph['x'][:10])
        #Mu=self.fa_mlp(Graph['x'])

        #print('aaa ', Graph['x'].shape)
        #print('bbb ', Mu.shape)
        Mu=Displace_node(Graph) 
        #print("Mu",Mu[:10])
        
        #print("Mu",Mu)
        #for k in range(Mu.shape[1]):
        #    plt.figure()
        #    plt.hist(Mu.detach().cpu()[:,k])
        #    plt.show()
        return Graph, node_probs, Mu 
    
    def Stride_optim(self,Graph,model,len_ep:int=10):
        """
        Graph : Batch of num_graphs
        len_ep : Optimization steps

        Return
        Rewards :(B_sz,len_ep)
        log_probs: (B_sz,len_ep)
        
        """
        B_sz=Graph.num_graphs
        log_length=len_ep
        #1: Create log dictionary
        log = {
        'Max_Mu': torch.zeros((B_sz, log_length,)),
        'Max_Disp': torch.zeros((B_sz, log_length,)),
        'Mean_Mu': torch.zeros((B_sz, log_length,)),
        'Mean_Disp': torch.zeros((B_sz, log_length,)),
        'Total_prob': torch.zeros((B_sz, log_length,)),
        'Reward': torch.zeros((B_sz, log_length,)),
        'd_PE': torch.zeros((B_sz, log_length,)),
        'PE': torch.zeros((B_sz, log_length,)),
        'States': []
        }
        #2: Initialize rewards and log-probs
        Rewards=torch.zeros((Graph.num_graphs,log_length))
        log_probs=torch.zeros((Graph.num_graphs,log_length))
        #log_probs=torch.ones((Graph.num_graphs,log_length))*(-1)


        #3: Run Adam baseline on same graphs
        Pos=Graph['pos'].clone()
        Adam_optim_pos=Batch_Adam_desc(0.01,100,Pos.reshape((Graph.num_graphs,-1,3)),self.Sys.Batch_Total_energy_fn,self.Sys.shift_fn)
        Adam_optim_energies=self.Sys.Batch_Total_energy_fn(Adam_optim_pos)
        
        #4 :Perform optimization
        for t in range(log_length):

            #A: Pass through the Policy_net model
            Out_G, node_probs, Mu =model(Graph)

            #B: Predict displacement from predicted mean using standard multivariate gaussian
            Pred_disp,disp_log_probs, disp_probs=pred_disp_vec(Mu,std=self.alpha)
            #Pred_disp=Mu
            
            #C: Displace nodes and update graph
            Pred_disp=torch.clip(Pred_disp, min=-self.disp_cutoff, max=self.disp_cutoff)
            New_Graph=self.Sys.update_G(Graph,Pred_disp)
            
            #D: Calculate rewards and log probs
            Rewards[:,t]=-(New_Graph['y']-Adam_optim_energies)#-Graph['y'])  #Reduction in energy
            log_probs[:,t]=torch.sum(disp_log_probs.reshape((Graph.num_graphs,-1)),dim=1)
            
            #E: Logs 
            Mu_magnitude=torch.sum(Mu.reshape((Graph.num_graphs,-1,3))**2,dim=2)
            Disp_magnitude=torch.sum(Pred_disp.reshape((Graph.num_graphs,-1,3))**2,dim=2)  #spatial dimension is 3
            log['Max_Mu'][:,t]=torch.sqrt(torch.max(Mu_magnitude,dim=1).values)
            log['Mean_Mu'][:,t]=torch.sqrt(torch.mean(Mu_magnitude,dim=1))
            log['Max_Disp'][:,t]=torch.sqrt(torch.max(Disp_magnitude,dim=1).values)
            log['Mean_Disp'][:,t]=torch.sqrt(torch.mean(Disp_magnitude,dim=1))
            log['Total_prob'][:,t]=torch.sum(disp_log_probs.reshape((Graph.num_graphs,-1))/1000,dim=1)
            log['d_PE'][:,t]=New_Graph['y']-Graph['y']
            log['Reward'][:,t]=Rewards[:,t]
            log['PE'][:,t]=Graph['y']
            log['States']+=[Graph['pos']]
            
            #F: Replace with new graph and continue optimization
            Graph=New_Graph
        return Graph, Rewards, log_probs , log


    def training_step(self,batch, batch_idx,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        Optim_G, Traj_Rewards, traj_log_probs,episode_log=self.Stride_optim(batch,self.forward,self.train_len_ep)
        d_PE=Optim_G['y']-batch['y']
        # Compute the loss and its gradients
        Loss_batch=Utils.Traj_Loss_fn(log_probs=traj_log_probs,Returns=get_discounted_returns(Rewards=Traj_Rewards,Y=0.95))
        loss=Loss_batch.mean() #Take mean of batch loss
        self.log("train/loss", loss,prog_bar=True)
        self.log("train/dPE",torch.mean(d_PE),prog_bar=True)
        return loss
    
    def validate_step(self,batch, batch_idx,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        Optim_G, Traj_Rewards, traj_log_probs,episode_log=self.Stride_optim(batch,self.forward,self.val_len_ep)
        d_PE=Optim_G['y']-batch['y']
        # Compute the loss and its gradients
        Loss_batch=Utils.Traj_Loss_fn(log_probs=traj_log_probs,Returns=get_discounted_returns(Rewards=Traj_Rewards,Y=0.95))
        loss=Loss_batch.mean() #Take mean of batch loss
        self.log("val/loss", loss,prog_bar=True)
        self.log("val/dPE",torch.mean(d_PE),prog_bar=True)
        return loss

    def test_step(self,batch, batch_idx,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        Optim_G, Traj_Rewards, traj_log_probs,episode_log=self.Stride_optim(batch,self.forward,self.val_len_ep)
        d_PE=Optim_G['y']-batch['y']
        # Compute the loss and its gradients
        Loss_batch=Utils.Traj_Loss_fn(log_probs=traj_log_probs,Returns=get_discounted_returns(Rewards=Traj_Rewards,Y=0.95))
        loss=Loss_batch.mean() #Take mean of batch loss
        self.log("test/loss", loss)
        self.log("test/dPE",torch.mean(d_PE))
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer