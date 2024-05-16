from pbdd.data_processing.dataset import DGT_Dataset
from pbdd.models.dgt_models import DGT_GAN,NECT_Generator,NECT_Discriminator
from pbdd.models.dgt_models import trainer_wrapper
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar

import os
import argparse
# load Dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator',type=str,default='NECT',choices=['NECT'])
    parser.add_argument('--discriminator',type=str,default='NECT',choices=['NECT','GAT'])
    parser.add_argument('--tqdm',action='store_true',default=False)
    parser.add_argument('--trainer_dir',type=str,\
                        help='directory for saving trainer')
    parser.add_argument('--data_path',type=str,help='directory for dataset')
    parser.add_argument('--d_threshold',type=float,default=0.55,\
                        help='threshold for training discriminator')
    parser.add_argument('--preset',type=int,default=0,choices=[0,1,2,3,4])
    parser.add_argument('--save_n_steps',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=48)
    parser.add_argument('--num_workers',type=int,default=12)
    parser.add_argument('--accelerator',type=str,default='auto')
    parser.add_argument('--devices',type=int,default=1)
    parser.add_argument('--strategy',type=str,default='ddp')
    parser.add_argument('--perturb',action='store_true',default=False)
    args = parser.parse_args()
    g_name = args.generator
    d_name = args.discriminator
    save_n_steps = args.save_n_steps
    batch_size = args.batch_size
    #num_workers = args.num_workers
    accelerator = args.accelerator
    devices = args.devices
    strategy = args.strategy

    trainer_dir = args.trainer_dir
    data_path = args.data_path
    print(args.perturb)
    if args.tqdm:
        tqdm_refresh_rate = 1
        enable_progress_bar = True
    else:
        tqdm_refresh_rate = 0
        enable_progress_bar = False
    
    # Load dataset
    assert os.path.exists(data_path), f'{data_path} does not exist'
    # data_path = '/fs/ess/PAA0203/wang12218/PBDD/data/Graph/G_v5_noH_max50/'
    G_data = DGT_Dataset(data_path)
    # G_data = LightningDataset(G_data)
    loader = DataLoader(G_data,batch_size=batch_size,shuffle=True,\
                        num_workers=args.num_workers,pin_memory=True,\
                        persistent_workers=True)

    # Define model and Trainer
    if g_name == 'NECT':
        if args.preset==0:
            generator = NECT_Generator(fn=1,fe=1,num_layers=6,\
                                       h1_channels=256,h2_channels=512,\
                                       hidden_node_dim=128,hidden_edge_dim=128,\
                                       out_node_channels=1,out_edge_channels=1,\
                                       use_pos=True)
        elif args.preset==1:
            # deep and narrow model is hard to train
            generator = NECT_Generator(fn=1,fe=1,num_layers=10,\
                                       h1_channels=128,h2_channels=256,\
                                       hidden_node_dim=64,hidden_edge_dim=64,\
                                       out_node_channels=1,out_edge_channels=1,\
                                       use_pos=True)
        elif args.preset==2:
            # add bottle neck layer is good for training
            generator = NECT_Generator(fn=1,fe=1,num_layers=6,\
                                       h1_channels=256,h2_channels=512,\
                                       hidden_node_dim=64,hidden_edge_dim=64,\
                                       out_node_channels=1,out_edge_channels=1,\
                                       use_pos=True)
        elif args.preset==3:
            # Modify preset 2
            generator = NECT_Generator(fn=1,fe=1,num_layers=7,\
                                       h1_channels=512,h2_channels=512,\
                                       hidden_node_dim=64,hidden_edge_dim=64,\
                                       out_node_channels=1,out_edge_channels=1,\
                                       use_pos=True)
        elif args.preset==4:
            # Large model setting, need more GPU memory or parallel training
            # Performance is not good
            generator = NECT_Generator(fn=1,fe=1,num_layers=6,\
                                       h1_channels=512,h2_channels=1024,\
                                       hidden_node_dim=256,hidden_edge_dim=256,\
                                       out_node_channels=1,out_edge_channels=1,\
                                       use_pos=True)
        use_pos_g = True
    if d_name == 'NECT':
        discriminator = NECT_Discriminator(fn=1,fe=1,num_layers=5,\
                                           h1_channels=128,h2_channels=256,\
                                           hidden_node_dim=128,hidden_edge_dim=128,\
                                           out_node_channels=256,use_pos=True)
        use_pos_d = True
    
    logger_name = f'{g_name}_{d_name}'
    model = DGT_GAN(generator,discriminator,\
                    use_pos_g=use_pos_g,use_pos_d=use_pos_d,\
                    d_threshold=args.d_threshold,perturb=args.perturb)
    print(model.perturb)
    # if args.gpus==1:
    trainer = trainer_wrapper(model,loader,5,trainer_dir,logger_name,\
                              log_every_n_steps=20,checkpoint_monitor=None,\
                              tqdm_refresh_rate=tqdm_refresh_rate,\
                              save_every_n_steps=save_n_steps,\
                              accelerator=accelerator,devices=devices,\
                              strategy=strategy,enable_progress_bar=enable_progress_bar)
    trainer.train()
