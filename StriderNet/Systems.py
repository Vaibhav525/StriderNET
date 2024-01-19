import torch
import numpy as np

class LennardJones_wbias:
    def __init__(self,species, box_size=4.736, sigma=None, epsilon=None, cutoffs=None,U0=10.0,U_sigma=1.25):
        self.box_size = box_size
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoffs = cutoffs
        self.pair_cutoffs = self.cutoffs[torch.meshgrid(species,species, indexing='xy')]
        self.pair_sigma = self.sigma[torch.meshgrid(species,species, indexing='xy')]
        self.pair_epsilon = self.epsilon[torch.meshgrid(species,species, indexing='xy')]
        self.U0=U0
        self.U_sigma=U_sigma
        self.Rms=None

    def periodic_shift(self, R, dR):
        """Shifts position and wraps them back within a periodic hypercube."""
        return torch.remainder(R + dR, self.box_size)

    def simple_displacement(self, Ra, Rb):
        return Ra - Rb

    def periodic_displacement(self, dR):
        return torch.remainder(dR + self.box_size * (0.5), self.box_size) - 0.5 * self.box_size

    def lj(self, r, sigma, eps, cutoff):
        within_cutoff =(r < cutoff)
        a = torch.pow((sigma / r), 6)
        return torch.where(within_cutoff, 4 * eps * a * torch.add(a, -1), 0.0)


    def append_Rm(self,Rm):
        if(self.Rms==None):
            self.Rms=Rm.unsqueeze(0)
        else:
            self.Rms=torch.cat((self.Rms,Rm.unsqueeze(0)),dim=0)
    def get_Rms(self):
        return self.Rms

    def bias_pot_single(self,R,Rm):
       r_rm_sq=torch.sum(self.displacement_fn(R,Rm)**2,dim=1)
       within_cutoff =(r_rm_sq < self.U_sigma**2)
       return self.U0*torch.sum(torch.where(within_cutoff, (1-(r_rm_sq)/(self.U_sigma**2))**2,0.0))

    def U_bias_tot(self,R):
        #U_bias_tot= Sum[k=0 to i-1]:U_bias_k   (U_bias_k depends on rm_k), i=no. of biases so far
        U_bias_tot_fn=torch.vmap(self.bias_pot_single,(None,0),0)
        U_bias_tot=U_bias_tot_fn(R,self.Rms)
        return torch.sum(U_bias_tot)
    
    def displacement_fn(self, Ra, Rb):
        dR = self.periodic_displacement(self.simple_displacement(Ra, Rb))
        return dR

    def shift_fn(self, R, dR):
        return self.periodic_shift(R, dR)

    def pair_dist_fn(self, R):
        dR = torch.vmap(torch.vmap(self.displacement_fn, (0, None), 0), (None, 0), 0)(R, R)
        Squared_dist = torch.sum(dR ** 2, axis=-1)
        Div_add=torch.eye(Squared_dist.shape[0]).to(Squared_dist)*1e10 #Large dist b/w self pairs
        Squared_dist=Squared_dist+Div_add
        dr = torch.sqrt(Squared_dist)
        return dr

    def pair_energies_fn(self, R):
        dr_pair = self.pair_dist_fn(R)
        E_pair = self.lj(dr_pair, self.pair_sigma, self.pair_epsilon, self.pair_cutoffs)
        return E_pair

    def node_energy_fn(self, R):
        return torch.sum(self.pair_energies_fn(R), dim=1)

    def total_energy_fn(self, R):
        return torch.sum(self.node_energy_fn(R)) * 0.5
    
    def total_energy_fn_wbias(self, R):
        if(self.Rms==None):
            return torch.sum(self.node_energy_fn(R)) * 0.5
        else:
            return torch.sum(self.node_energy_fn(R)) * 0.5 + self.U_bias_tot(R)
    
    def get_functions(self):
        return self.displacement_fn, self.pair_dist_fn, self.node_energy_fn, self.total_energy_fn,self.total_energy_fn_wbias, self.shift_fn, self.pair_cutoffs, self.pair_sigma, self.pair_epsilon


# Example usage
# Now you can use these functions as needed
# sigma = torch.Tensor([[1.0, 0.8], [0.8, 0.88]])
# epsilon = torch.Tensor([[1.0, 1.5], [1.5, 0.5]])
# cutoffs = torch.Tensor([[1.5, 1.25], [1.25, 2.0]])
# box_size=22.0464
# lj_calculator = LennardJones_wbias(species=species,box_size=box_size,sigma=sigma, epsilon=epsilon, cutoffs=cutoffs)
# displacement_fn, pair_dist_fn, node_energy_fn, total_energy_fn,total_energy_fn_wbias, shift_fn, pair_cutoffs, pair_sigma, pair_epsilon = lj_calculator.get_functions()




def periodic_shift(side , R, dR):
  """Shifts position and wraps them back within a periodic hypercube."""
  return torch.remainder(R + dR, side)

def pairwise_displacement(Ra, Rb):
    return Ra-Rb

def periodic_displacement(side, dR):
    return torch.remainder(dR + side * (0.5), side) - 0.5 * side


def lennard_jones(species, 
                box_size :float=4.3676,
                sigma:torch.Tensor =torch.Tensor([[1.0 ,0.8],[0.8 ,0.88]]),
                epsilon:torch.Tensor =torch.Tensor([[1.0 ,1.5],[1.5 ,0.5]]),
                cutoffs:torch.Tensor =torch.Tensor([[2.5 ,2.0],[2.0 ,2.2]])
            ):
    
    #2 : Create box
    def lj(r :torch.Tensor,sigma,eps,cutoff):
        """
        -Calculatrs lennard jones potential energy
        Args:
        r: radial distance between pairs of atoms (1,)
        sigma: sigma parameter (1,)
        epsilon: epsilon parameter (1,)
        cutoff : zero energy beyond cutoff
        Returns:
        LJ potential (1,)
        """
        cutoff=cutoff.to(r)
        sigma=sigma.to(r)
        eps=eps.to(r)
        within_cutoff = (r < cutoff)
        a=torch.pow((sigma/r),6)
        return torch.where(within_cutoff, 4*eps*a*torch.add(a,-1),0.0)

    
    
    def displacement_fn(Ra, Rb):
        dR = periodic_displacement(box_size, pairwise_displacement(Ra, Rb))
        return dR
    def shift_fn(R, dR):
      return periodic_shift(box_size, R, dR)
    
    #4 : Create random configuration
    
    #Calculating pair distances
    Disp_Vec_fn= torch.vmap(torch.vmap(displacement_fn, (0, None), 0), (None, 0), 0)
    def pair_dist_fn(R):
        dR = Disp_Vec_fn(R, R)
        Squared_dist=torch.sum(dR ** 2, axis=-1)
        Div_add=torch.eye(Squared_dist.shape[0]).to(Squared_dist)*1e10 #Large dist b/w self pairs
        Squared_dist=Squared_dist+Div_add
        #print("dR",torch.sum(dR ** 2, axis=-1))
        dr = torch.sqrt(Squared_dist)
        return dr
    
    #5: Creating neigh_list and senders and receivers
    pair_cutoffs=cutoffs[torch.meshgrid(species,species,indexing='xy')]#matrix_broadcast_fn(cutoffs,species,species)
    pair_sigma=sigma[torch.meshgrid(species,species,indexing='xy')]#matrix_broadcast_fn(sigma,species,species)
    pair_epsilon=epsilon[torch.meshgrid(species,species,indexing='xy')]#(epsilon,species,species)
        
    #6: Calculate pair energies, node energies
    def pair_energies_fn(R):
        dr_pair = pair_dist_fn(R)
        E_pair=lj(dr_pair,pair_sigma,pair_epsilon,pair_cutoffs)
        return E_pair
    
    def Node_energy_fn(R):
        return torch.sum(pair_energies_fn(R),dim=1)
    
    def Total_energy_fn(R):
        return torch.sum(Node_energy_fn(R))*0.5
    return Disp_Vec_fn,pair_dist_fn,Node_energy_fn,Total_energy_fn,displacement_fn, shift_fn ,pair_cutoffs,pair_sigma,pair_epsilon
    
