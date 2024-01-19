import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mv_normal
from  torch.distributions import multivariate_normal
import os
import yaml


def pdf_multivariate_gauss(x, mu, cov):
    """ pdf gives density not probability. Thus we remove part1 for scaling  [w/0 part1 output is within 0 to 1]"""
   
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    #assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    #assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    #assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    #assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    #assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    #part1 = 1 / ( ((2* jnp.pi)**(len(mu)/2)) * (jnp.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.matmul(torch.linalg.inv(cov))).matmul((x-mu))

    
    #return part1 * jnp.exp(part2)
    return torch.exp(part2)

vmap_pdf_multivariate_gauss=torch.vmap(pdf_multivariate_gauss,(0,0,None))


def pred_disp_vec(Mu,std=0.01):
    mean=torch.zeros_like(Mu).to(Mu)
    cov =torch.eye(mean.shape[1]).to(Mu)*(std**2)
    Mv_dist= torch.distributions.multivariate_normal.MultivariateNormal(mean,cov)
    Pred=Mv_dist.sample()
    Pred_disp=Mu+Pred
    #Pred_disp=Mu
    probs=vmap_pdf_multivariate_gauss(Pred,mean,cov)

    #dist = multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov)
    # probs = torch.exp(dist.log_prob(Pred_disp))

    return Pred_disp,torch.log(probs), probs

def get_discounted_returns(Rewards, Y=0.95):
    """Calculates discounted rewards"""
    res = torch.zeros_like(Rewards)
    Temp_G = torch.zeros((Rewards.shape[0],))
    for k in range(Rewards.shape[1]-1, -1, -1):
        Temp_G = Rewards[:, k] + Y * Temp_G
        res[:, k] = Temp_G
    return res



def print_log(log,epoch_id=0,Batch_id=0):
    B_sz=log['Reward'].shape[0]
    log_length=log['Reward'].shape[1]
    for k in range(B_sz):
        print("\n#GraphNo. ",k+1)
        print("\nStep\tMax_Mu\tMean_Mu\t   Max_Disp  Mean_Disp\tLog_Total_prob\tReward\t d_PE\t  PE")
        for i in range(log['Reward'].shape[1]):
            print(i+1,"\t%8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f"%(log['Max_Mu'][k][i],log['Mean_Mu'][k][i],log['Max_Disp'][k][i],log['Mean_Disp'][k][i],log['Total_prob'][k][i],log['Reward'][k][i],log['d_PE'][k][i],log['PE'][k][i]))


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def write_yaml_config(config, output_folder, output_file_name):
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, output_file_name)
    with open(output_file_path, 'w') as output_file:
        yaml.dump(config, output_file, default_flow_style=False)

def Traj_Loss_fn(*, log_probs, Returns):
    return torch.sum(log_probs * Returns, dim=1)