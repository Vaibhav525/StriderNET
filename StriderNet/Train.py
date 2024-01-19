import torch
import torch_geometric
import numpy as np

from Systems import lennard_jones
import Systems
from Generic_system import Generic_system
from Utils import *
import Utils
from Optimizers import *
import argparse
import random
import time
import wandb
import matplotlib.pyplot as plt
import os
import yaml
import pytorch_lightning as pl
from models import *
from pytorch_lightning.callbacks import ModelCheckpoint, Timer ,ModelSummary ,EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger



def main(config):
    torch.manual_seed(42)  # Setting the seed
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    Sys = Generic_system()
    Train_dataloader, Test_dataloader, Val_dataloader = Sys.create_batched_States(System='LJ', Batch_size=config['b_sz'], N_sample=config['N_Sample'], Traj_path=config['data_dir'],Species_path=config['species_dir'])

    # Initialize model
    Init_batch = next(iter(Train_dataloader))
    model = Pol_Net_lit(
        sys_env=Sys,
        in_edge_feats=Init_batch['edge_attr'].shape[1],
        in_node_feats=Init_batch['x'].shape[1],
        node_emb_size=config['node_emb_size'],
        edge_emb_size=config['edge_emb_size'],
        fa_layers=config['fa_layers'],
        fb_layers=config['fb_layers'],
        fe_layers=config['fe_layers'],
        fv1_layers=config['fv_layers'],
        fv2_layers=config['fv_layers'],
        MLP1_layers=config['MLP1_layers'],
        MLP2_layers=config['MLP2_layers'],
        sigma=config['sigma'],
        multivariate_std=config['alpha'],
        disp_cutoff=config['disp_cutoff'],
        train_len_ep=config['train_len_ep'],
        val_len_ep=config['val_len_ep'],
        batchnorm_running_stats=True,
        message_passing_steps=config['message_passing_steps']
    )


    wandb_logger = WandbLogger(project="my-awesome-project",name='Train_lit1',save_dir=config['out_dir'],offline=config['wandb_offline'])
    wandb_logger.watch(model)
    pl.seed_everything(config['seed'],workers=True)
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                        logger=wandb_logger,
                        log_every_n_steps=config['log_step_freq'],
                        accumulate_grad_batches=config['grad_acc'],
                        benchmark=config['benchmark_flag'],
                        deterministic=config['is_deterministic'],
                        check_val_every_n_epoch=config['val_freq'],
                        callbacks=[ModelCheckpoint(monitor="val_loss"),
                                    Timer(duration=config['max_duration']),
                                    ModelSummary(max_depth=2),
                                    EarlyStopping(monitor='val/dPE',divergence_threshold=1000),
                                    StochasticWeightAveraging(swa_lrs=1e-3)]
                        )

    #More arguments for Trainer:
    #profiler='simple'|'advanced'  "To check time taken by each function"


    trainer.fit(model=model, train_dataloaders=Train_dataloader)
    torch.save(model.state_dict(), config['out_dir'] + "Models/" + 'Final_torch_model')
    Utils.write_yaml_config(config, config['out_dir'], 'Models/Params_config.yaml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StriderNET Torch arguments")
    parser.add_argument('--config', type=str, default='Stridernet_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    config = load_config(args.config)

    # Write the YAML config to Output folder for future reference
    write_yaml_config(config, config['out_dir'], 'In_config.yaml')

    
    main(config)

     