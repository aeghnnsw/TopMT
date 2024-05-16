# Models for Deep Graph Translation
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch_geometric
from torch_geometric.nn import aggr
from torch_geometric.nn import MessagePassing,GraphNorm,BatchNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.models import GAT
from torch_geometric.utils import subgraph,to_undirected
from typing import Any, Dict, Optional
from torch.nn import Linear, Sequential, ReLU, Sigmoid,PReLU
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pbdd.data_processing.utils import (strip_mol_graph,
                                        draw_graph,
                                        draw_graph_with_type,
                                        perturb_graph,
                                        perturb_graph_type)

import os
import numpy as np
from glob import glob

class NECT_layer(MessagePassing):
    def __init__(self,fn,fe,h1_channels,h2_channels,\
                 out_node_channels,out_edge_channels,\
                 use_pos:bool=False,dropout:float=0.0):
        super().__init__(aggr=aggr.MultiAggregation(['add','mean','std']))
        self.use_pos = use_pos
        self.fn = fn
        self.fe = fe
        if use_pos==False:
            self.LN11 = Linear(2*fn+fe,h1_channels,bias=False)
            self.LE11 = Linear(2*fn+fe,h1_channels,bias=False)
        else:
            self.LN11 = Linear(2*fn+fe+3,h1_channels,bias=False) 
            self.LE11 = Linear(2*fn+fe+3,h1_channels,bias=False)
        self.LN12 = Linear(h1_channels,h2_channels,bias=False)
        self.LN2 = Linear(3*h2_channels,out_node_channels,bias=False)
        self.prelu_n11 = PReLU()
        # self.dropout_n11 = torch.nn.Dropout(dropout)
        self.prelu_n12 = PReLU()
        self.prelu_n2 = PReLU()
        

        self.LE12 = Linear(h1_channels,h2_channels,bias=False)
        self.LE2 = Linear(2*3*h2_channels,out_edge_channels,bias=False)
        self.prelu_e11 = PReLU()
        # self.dropout_e11 = torch.nn.Dropout(dropout)
        self.prelu_e12 = PReLU()
        self.prelu_e2 = PReLU()

    def forward(self,x,edge_index,edge_attr,pos:Optional[torch.Tensor]=None):
        # edge_index = edge_index.to(self.cpu)
        # print(edge_index.device) 
        # print('x device: ',x.device)    
        edge_effect_nn1 = x[edge_index[0]]
        edge_effect_nn2 = x[edge_index[1]]
        edge_effect_ne = edge_attr
        if self.use_pos:
            deltax = pos[edge_index[0],:]-pos[edge_index[1],:]
            # print(f'edge_effect_nn1 device: {edge_effect_nn1.device}')
            # print(f'edge_effect_nn2 device: {edge_effect_nn2.device}')
            # print(f'edge_effect_ne device: {edge_effect_ne.device}')
            # print(f'deltax device: {deltax.device}')
            edge_effect_concat = torch.cat((edge_effect_nn1,edge_effect_nn2,edge_effect_ne,deltax),dim=1)
        else:
            edge_effect_concat = torch.cat((edge_effect_nn1,edge_effect_nn2,edge_effect_ne),dim=1)
        edge_effect_node = self.LN11(edge_effect_concat)
        edge_effect_node = self.prelu_n11(edge_effect_node)
        # edge_effect_node = self.dropout_n11(edge_effect_node)
        edge_effect_node = self.LN12(edge_effect_node)
        edge_effect_node = self.prelu_n12(edge_effect_node)
        node_effect_node = self.propagate(edge_index,x=x,edge_effect=edge_effect_node)
        node_x_update = self.LN2(node_effect_node)
        node_x_update = self.prelu_n2(node_x_update)

        # Edge Update
        edge_effect_edge = self.LE11(edge_effect_concat)
        edge_effect_edge = self.prelu_e11(edge_effect_edge)
        # edge_effect_edge = self.dropout_e11(edge_effect_edge)
        edge_effect_edge = self.LE12(edge_effect_edge)
        edge_effect_edge = self.prelu_e12(edge_effect_edge)
        node_effect_edge = self.propagate(edge_index,x=x,edge_effect=edge_effect_edge)
        edge_attr_new = torch.cat((node_effect_edge[edge_index[0]],node_effect_edge[edge_index[1]]),dim=1)
        edge_attr_update = self.LE2(edge_attr_new)
        edge_attr_update = self.prelu_e2(edge_attr_update)
        return node_x_update,edge_attr_update

    def message(self,x_j,edge_effect):
        return edge_effect


class NECT_Generator(LightningModule):
    # Generator for Deep Graph Translation
    def __init__(self,fn:int,fe:int,num_layers:int,\
                 h1_channels:int,h2_channels:int,\
                 hidden_node_dim:int,hidden_edge_dim:int,\
                 out_node_channels:int,out_edge_channels:int,\
                 use_pos:bool=True,dropout:float=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.use_pos = use_pos
        self.layers = torch.nn.ModuleList()
        self.graph_norm_layer = torch.nn.ModuleList()
        self.edge_batch_norm_layer = torch.nn.ModuleList()
        fn1 = fn
        fe1 = fe
        for i in range(num_layers-1):
            self.layers.append(NECT_layer(fn1,fe1,h1_channels,h2_channels,\
                                          hidden_node_dim,hidden_edge_dim,\
                                          use_pos=use_pos,dropout=dropout))
            fn1 = hidden_node_dim
            fe1 = hidden_edge_dim
            self.graph_norm_layer.append(GraphNorm(hidden_node_dim))
            self.edge_batch_norm_layer.append(BatchNorm(hidden_edge_dim))
        self.layers.append(NECT_layer(fn1,fe1,h1_channels,h2_channels,\
                                      out_node_channels,out_edge_channels,\
                                      use_pos=use_pos,dropout=dropout))
        self.graph_norm_layer.append(GraphNorm(out_node_channels))
        self.edge_batch_norm_layer.append(BatchNorm(out_edge_channels))
                                      

    def forward(self,x,edge_index,edge_attr,batch,pos:Optional[torch.Tensor]=None):
        for i in range(self.num_layers):
            x,edge_attr = self.layers[i](x,edge_index,edge_attr,pos=pos)
            x = self.graph_norm_layer[i](x,batch)
            edge_attr = self.edge_batch_norm_layer[i](edge_attr)
        _,edge_attr = to_undirected(edge_index,edge_attr,reduce='mean')
        return x,edge_attr


class GAT2_Discriminator(LightningModule):
    # Use GAT as discriminator
    def __init__(self,fn:int,fe:int,hidden_channels,out_channels,num_layers:int):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.gat = GAT(fn,hidden_channels,num_layers,out_channels=out_channels,v2=True,edge_dim=fe)
        self.mlp = Sequential(Linear(3*out_channels,3*out_channels//2),ReLU(),\
                              Linear(3*out_channels//2,3*out_channels//4),ReLU(),\
                              Linear(3*out_channels//4,1))
    def forward(self,x,edge_index,edge_attr,batch):
        x = self.gat(x,edge_index,edge_attr=edge_attr)
        x1 = global_add_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        x3 = global_max_pool(x,batch)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.mlp(x)
        return x
    
class NECT_Discriminator(LightningModule):
    def __init__(self,fn:int,fe:int,num_layers:int,\
                 h1_channels:int,h2_channels:int,\
                 hidden_node_dim:int,hidden_edge_dim:int,\
                 out_node_channels:int,\
                 use_pos:bool=True):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.use_pos = use_pos
        self.layers = torch.nn.ModuleList()
        self.graph_norm_layer = torch.nn.ModuleList()
        self.edge_batch_norm_layer = torch.nn.ModuleList()
        fn1 = fn
        fe1 = fe
        for i in range(num_layers-1):
            self.layers.append(NECT_layer(fn1,fe1,h1_channels,h2_channels,\
                                          hidden_node_dim,hidden_edge_dim,\
                                          use_pos=use_pos))
            fn1 = hidden_node_dim
            fe1 = hidden_edge_dim
            self.graph_norm_layer.append(GraphNorm(hidden_node_dim))
            self.edge_batch_norm_layer.append(BatchNorm(hidden_edge_dim))
        self.layers.append(NECT_layer(fn1,fe1,h1_channels,h2_channels,\
                                        out_node_channels,1,use_pos=use_pos))
        self.graph_norm_layer.append(GraphNorm(out_node_channels))
        self.edge_batch_norm_layer.append(BatchNorm(1))
        self.mlp = Sequential(Linear(3*out_node_channels,3*out_node_channels//2),PReLU(),\
                              Linear(3*out_node_channels//2,3*out_node_channels//4),PReLU(),\
                              Linear(3*out_node_channels//4,1))
    def forward(self,x,edge_index,edge_attr,batch,pos:Optional[torch.Tensor]=None):
        for i in range(self.num_layers):
            x,edge_attr = self.layers[i](x,edge_index,edge_attr,pos=pos)
            x = self.graph_norm_layer[i](x,batch)
            edge_attr = self.edge_batch_norm_layer[i](edge_attr)
        x1 = global_add_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        x3 = global_max_pool(x,batch)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.mlp(x)
        return x


class DGT_Discriminator(LightningModule):
    def __init__(self,discriminator:torch.nn.Module,alpha:float=0.5,perturb_mode:int=0,\
                 plot_every_step:int=100,use_pos:bool=False):
        super().__init__()
        self.discriminator = discriminator
        assert alpha>=0 and alpha<=1,'alpha must be in [0,1]'
        assert perturb_mode in [0,1,2],'perturb_mode must be in [0,1,2]'
        self.alpha = alpha
        self.perturb_mode = perturb_mode
        self.cpu_device = torch.device('cpu')
        self.plot_every_step = plot_every_step
        self.use_pos = use_pos

    def forward(self,x,edge_index,edge_attr,batch,**kwargs):
        return self.discriminator(x,edge_index,edge_attr,batch,**kwargs)
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy_with_logits(y_hat,y)
    
    def training_step(self,batch,batch_idx):
        x_r = (batch.y[:,0]>0).float().unsqueeze(1)
        x_r = x_r + torch.randn_like(x_r)*0.1
        x_r = x_r.clamp(0,1)
        edge_attr_r = (batch.edge_attr[:,1]>0).float().unsqueeze(1)
        edge_attr_r = edge_attr_r + torch.randn_like(edge_attr_r)*0.1
        edge_attr_r = edge_attr_r.clamp(0,1)
        edge_attr_p,x_p = perturb_graph(batch.edge_index,edge_attr_r,x_r,\
                                        alpha=self.alpha,mode=self.perturb_mode)
        # plot real graph and purturbed graph
        if self.global_step%self.plot_every_step==0:
            subset0 = torch.arange(batch.ptr[0],batch.ptr[1])
            edge_index_r0,edge_attr_r0 = subgraph(subset0,batch.edge_index.to(self.cpu_device),\
                                                  edge_attr=edge_attr_r.to(self.cpu_device),\
                                                  relabel_nodes=False)
            edge_index_p0,edge_attr_p0 = subgraph(subset0,batch.edge_index.to(self.cpu_device),\
                                                  edge_attr=edge_attr_p.to(self.cpu_device),\
                                                  relabel_nodes=False)
            r_temp = torch_geometric.data.Data(x=x_r[subset0].to(self.cpu_device),\
                                               edge_index=edge_index_r0,\
                                               edge_attr=edge_attr_r0)
            r_temp = strip_mol_graph(r_temp,r_temp.edge_attr)
            p_temp = torch_geometric.data.Data(x=x_p[subset0].to(self.cpu_device),\
                                               edge_index=edge_index_p0,\
                                               edge_attr=edge_attr_p0)
            p_temp = strip_mol_graph(p_temp,p_temp.edge_attr)
            np_fig_r = draw_graph(r_temp,show=False,return_np=True,color='steelblue')
            np_fig_p = draw_graph(p_temp,show=False,return_np=True,color='firebrick')
            self.logger.experiment.add_image('real_graph',np_fig_r,self.global_step)
            self.logger.experiment.add_image('perturbed_graph',np_fig_p,self.global_step)
        # train discriminator
        if self.use_pos:       
            r_pred = self.discriminator(x_r,batch.edge_index,edge_attr_r,batch.batch,pos=batch.pos)
            p_pred = self.discriminator(x_p,batch.edge_index,edge_attr_p,batch.batch,pos=batch.pos)
        else:
            r_pred = self.discriminator(x_r,batch.edge_index,edge_attr_r,batch.batch)
            p_pred = self.discriminator(x_p,batch.edge_index,edge_attr_p,batch.batch)
        valid = torch.ones_like(r_pred)
        fake = torch.zeros_like(p_pred)
        real_loss = self.adversarial_loss(r_pred,valid)
        fake_loss = self.adversarial_loss(p_pred,fake)
        d_loss = (real_loss+fake_loss)/2
        self.log('real_loss',real_loss,prog_bar=True)
        self.log('fake_loss',fake_loss,prog_bar=True)
        self.log('d_loss',d_loss,prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=2e-4)
        return optimizer

class DGT_GAN(LightningModule):
    def __init__(self,generator:LightningModule,discriminator:LightningModule,\
                 use_pos_g:bool=True,use_pos_d:bool=False,d_threshold:float=0.5,\
                 perturb:bool=False):
        super().__init__()
        self.save_hyperparameters(ignore=['generator','discriminator'])
        # self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator

        self.cpu_device = torch.device('cpu')
        self.use_pos_g = use_pos_g
        self.use_pos_d = use_pos_d
        self.d_threshold = d_threshold
        self.perturb = perturb

    def forward(self,z,edge_index,edge_attr,batch,pos:Optional[torch.Tensor]=None):
        x,edge_attr = self.generator(z,edge_index,edge_attr,batch,pos=pos)
        x = (torch.tanh(x) + 1)/2
        edge_attr = torch.tanh(edge_attr) + 1 + 1e-6
        edge_attr = edge_attr/torch.max(edge_attr)
        _,edge_attr=to_undirected(edge_index,edge_attr,reduce='mean')
        return x,edge_attr

    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy_with_logits(y_hat,y)
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        # batch is graph data type
        #   x: node feature [x,y,z,degree]
        #   edge_index: edge index
        #   edge_attr: edge feature [1,edge_type(0,or >0)]
        #   y: node label [simplified_atom_type,real_degree]
        #   pos: node position [x,y,z]
        #   use: Ultrafast Shape Recognition Descriptor 12 dimensions

        # sample noise
        z = torch.randn(batch.x.shape[0],1).type_as(batch.x)
        edge_z = torch.ones(batch.edge_attr.shape[0],1).type_as(batch.edge_attr)
        # generate graphs
        x_g,edge_attr_g = self(z,batch.edge_index,edge_z,batch.batch,batch.pos)

        # Train discriminator for 1 epoch before training generator

        # train generator
        if optimizer_idx == 0:            
            if batch_idx % 10 == 0:
                # random generate multiple graphs
                subset0 = torch.arange(batch.ptr[0],batch.ptr[1])
                z_temp = torch.randn(batch.x.shape[0],1).type_as(batch.x)
                edge_z_temp = torch.ones(batch.edge_attr.shape[0],1).type_as(batch.edge_attr)
                x_g_temp,edge_attr_g_temp = self(z_temp,batch.edge_index,edge_z_temp,batch.batch,batch.pos)
                edge_idex_g0,edge_attr_g0 = subgraph(subset0,batch.edge_index.to(self.cpu_device),\
                                                        edge_attr=edge_attr_g_temp.to(self.cpu_device),\
                                                        relabel_nodes=False)
                g_temp = torch_geometric.data.Data(x=x_g[subset0].to(self.cpu_device),\
                                                edge_index=edge_idex_g0,\
                                                edge_attr=edge_attr_g0) 
                g_temp = strip_mol_graph(g_temp,g_temp.edge_attr)
                np_fig = draw_graph(g_temp,show=False,return_np=True,color='firebrick')
                self.logger.experiment.add_image(f'generated_graph',np_fig,batch_idx)
            # if self.global_step % 2 == 0:
            if self.use_pos_g:
                g_pred = self.discriminator(x_g,batch.edge_index,edge_attr_g,\
                                            batch.batch,pos=batch.pos)
            else:
                g_pred = self.discriminator(x_g,batch.edge_index,edge_attr_g,batch.batch)
            valid = torch.ones_like(g_pred)
            g_loss = self.adversarial_loss(g_pred,valid)
            self.log('g_loss',g_loss,prog_bar=True)

            # else:
                # g_loss = None
            # plot generated graph
            return g_loss
        
        # train discriminator
        if optimizer_idx == 1:
            edge_attr_r = (batch.edge_attr[:,1]>0).float().unsqueeze(1)
            edge_attr_r = edge_attr_r + torch.randn_like(edge_attr_r)*0.1
            edge_attr_r = edge_attr_r.clamp(0,1)
            n_edge_r = torch.sum(edge_attr_r>0.5).item()
            n_edge_g = torch.sum(edge_attr_g>0.5).item()
            x_r = (batch.y[:,0]>0).float().unsqueeze(1)
            x_r = x_r + torch.randn_like(x_r)*0.2
            x_r = x_r.clamp(0,1)
            
            if self.use_pos_d:
                r_pred = self.discriminator(x_r,batch.edge_index,edge_attr_r,\
                                            batch.batch,pos=batch.pos)
                g_pred = self.discriminator(x_g.detach(),batch.edge_index,edge_attr_g.detach(),\
                                            batch.batch,pos=batch.pos)
            else:
                r_pred = self.discriminator(x_r,batch.edge_index,edge_attr_r,batch.batch)
                g_pred = self.discriminator(x_g.detach(),batch.edge_index,edge_attr_g.detach(),batch.batch)
                
            valid = torch.ones_like(r_pred)
            fake_g = torch.zeros_like(r_pred)
            real_loss = self.adversarial_loss(r_pred,valid)
            fake_loss_g = self.adversarial_loss(g_pred,fake_g)
            
            # perturb graph
            if self.perturb:
                alpha = np.random.uniform(0.05,0.95)
                mode = np.random.randint(3)
                edge_attr_p,x_p = perturb_graph(batch.edge_index,edge_attr_r,x_r,\
                                                alpha,mode)
                if self.use_pos_d:
                    p_pred = self.discriminator(x_p,batch.edge_index,edge_attr_p,\
                                                batch.batch,pos=batch.pos)
                else:
                    p_pred = self.discriminator(x_p,batch.edge_index,edge_attr_p,batch.batch)
                fake_p = torch.zeros_like(p_pred)
                fake_loss_p = self.adversarial_loss(p_pred,fake_p)
                self.log('fake_loss_p',fake_loss_p,prog_bar=True)
                d_loss = (2*real_loss + fake_loss_g + fake_loss_p)/4
            else:
                d_loss = (real_loss + fake_loss_g)/2
            self.log('real_loss',real_loss,prog_bar=True)
            self.log('fake_loss_g',fake_loss_g,prog_bar=True)
            self.log('d_loss',d_loss,prog_bar=True)
            self.log('n_edges_ratio',n_edge_g/n_edge_r,prog_bar=True)
            # plot real graph
            if batch_idx % 10 == 0:
                subset0 = torch.arange(batch.ptr[0],batch.ptr[1])
                edge_idex_r0,edge_attr_r0 = subgraph(subset0,batch.edge_index.to(self.cpu_device),\
                                                     edge_attr=edge_attr_r.to(self.cpu_device),\
                                                     relabel_nodes=False)
                r_temp = torch_geometric.data.Data(x=x_r[subset0].to(self.cpu_device),\
                                                   edge_index=edge_idex_r0,\
                                                   edge_attr=edge_attr_r0)
                r_temp = strip_mol_graph(r_temp,r_temp.edge_attr)
                np_fig = draw_graph(r_temp,show=False,return_np=True,color='steelblue')
                self.logger.experiment.add_image('real_graph',np_fig,batch_idx)
            # stop train discriminator if discriminator is too strong
            if (real_loss+fake_loss_g)/2 < self.d_threshold and real_loss<0.69 and fake_loss_g<0.69:
                return d_loss * 0
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),lr=0.0002,betas=(0.5,0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
        return [opt_g,opt_d],[]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['g_hparams'] = self.generator.hparams
        checkpoint['d_hparams'] = self.discriminator.hparams
        # return super().on_save_checkpoint(checkpoint)

    @classmethod
    def load_GAN_from_checkpoint(cls,checkpoint_path:str,map_location=torch.device('cpu')):
        # if checkpoint_path is a file, directly load it
        # if checkpoint_path is a directory, load the latest checkpoint
        assert os.path.exists(checkpoint_path),'checkpoint_path does not exist'
        if os.path.isdir(checkpoint_path):
            ckpt_files = os.path.join(checkpoint_path,'*.ckpt')
            ckpt_paths = glob(ckpt_files)
            if len(ckpt_paths)==0:
                raise ValueError(f'No checkpoint found in {checkpoint_path}')
            ckpt_paths.sort(key=os.path.getmtime,reverse=True)
            checkpoint_path = ckpt_paths[0]
        params = torch.load(checkpoint_path,map_location=map_location)
        g_hparams = params['g_hparams']
        d_hparams = params['d_hparams']
        generator = NECT_Generator(**g_hparams)
        discriminator = NECT_Discriminator(**d_hparams)
        model = DGT_GAN(generator,discriminator)
        model.load_state_dict(params['state_dict'])
        return model

# class type_pred_GAN(LightningModule):
#     def __init__(self,generator:LightningModule,discriminator:LightningModule,\
#                  d_threshold:float=0.1,perturb:bool=False,warmup_steps:int=1000,\
#                  ele_loss:bool=False):
#         super().__init__()
#         self.save_hyperparameters(ignore=['generator','discriminator'])
#         self.generator = generator
#         self.discriminator = discriminator
#         self.cpu_device = torch.device('cpu')
#         self.d_threshold = d_threshold
#         self.perturb = perturb
#         self.warmup_steps = warmup_steps
#         self.ele_loss = ele_loss
    
#     def forward(self,z,edge_index,edge_attr,batch):
#         x,edge_out = self.generator(z,edge_index,edge_attr,batch)
#         # perform softmax on node and edge
#         x = torch.softmax(x,dim=1)
#         edge_out = torch.softmax(edge_out,dim=1)
#         return x,edge_out
    
#     def adversarial_loss(self,y_hat,y):
#         return F.binary_cross_entropy_with_logits(y_hat,y)
    
#     def element_loss(self,y_hat,y):
#         return F.mse_loss(y_hat,y)
    
#     def training_step(self,batch,batch_idx,optimizer_idx):
#         # sample noise
#         z = torch.randn(batch.x.shape[0],1).type_as(batch.x)
#         edge_z = torch.randn(batch.edge_attr.shape[0],1).type_as(batch.x)
#         # generate node type and edge type
#         atom_g,edge_g = self(z,batch.edge_index,edge_z,batch.batch)
#         # train generator
#         if optimizer_idx == 0:
#             g_pred = self.discriminator(atom_g,batch.edge_index,edge_g,batch.batch)
#             valid = torch.ones_like(g_pred)
#             g_loss = self.adversarial_loss(g_pred,valid)
#             self.log('g_loss',g_loss,prog_bar=True)


#             # log images, generate two images to check mode collapse
#             if batch_idx % 100 == 0:
#                 sample_data = batch.to_data_list()[0]
#                 for i in range(5):
#                     batch_temp = torch.zeros(sample_data.x.shape[0]).type_as(batch.batch)
#                     z_temp = torch.randn(sample_data.x.shape[0],1).type_as(sample_data.x)
#                     edge_z_temp = torch.randn(sample_data.edge_attr.shape[0],1).type_as(sample_data.x)
#                     atom_g_temp,edge_g_temp = self(z_temp,sample_data.edge_index,\
#                                                    edge_z_temp,batch_temp)
#                     g_temp = torch_geometric.data.Data(x=atom_g_temp.to(self.cpu_device),\
#                                                        edge_index=sample_data.edge_index,\
#                                                        edge_attr=edge_g_temp.to(self.cpu_device))
#                     np_fig = draw_graph_with_type(g_temp,show=False,return_np=True)
#                     self.logger.experiment.add_image(f'generated_graph{i}',np_fig,batch_idx)
#             if self.global_step<self.warmup_steps:
#                 g_loss = g_loss * 0
#             else:
#                 # element_loss
#                 if self.element_loss:
#                     real_r_mean = torch.mean(batch.x,dim=0)
#                     atom_g_mean = torch.mean(torch.softmax(5*atom_g,dim=1),dim=0)
#                     ele_loss = self.element_loss(atom_g_mean,real_r_mean)
#                     self.log('ele_loss',ele_loss,prog_bar=True)
#                     g_loss = g_loss + 0.5*ele_loss
#             return g_loss
#         # train discriminator
#         if optimizer_idx == 1:
#             # perturb atom and bond type

#             atom_r = 3*batch.x + 0.1*torch.randn_like(batch.x)
#             atom_r = torch.softmax(atom_r,dim=1)
#             edge_r = 2*batch.edge_attr + 0.2*torch.randn_like(batch.edge_attr)
#             edge_r = torch.softmax(edge_r,dim=1)
#             r_pred = self.discriminator(atom_r,batch.edge_index,edge_r,batch.batch)
#             g_pred = self.discriminator(atom_g.detach(),batch.edge_index,edge_g.detach(),batch.batch)
#             valid = torch.ones_like(r_pred)
#             fake_g = torch.zeros_like(r_pred)
#             real_loss = self.adversarial_loss(r_pred,valid)
#             fake_loss_g = self.adversarial_loss(g_pred,fake_g)
#             if self.perturb:
#                 alpha = np.random.uniform(0.2,0.9)
#                 atom_p,edge_p = perturb_graph_type(batch.edge_index,batch.x,batch.edge_attr,alpha)
#                 atom_p = 3*batch.x + 0.2*torch.randn_like(batch.x)
#                 atom_p = torch.softmax(atom_p,dim=1)
#                 edge_p = 2*batch.edge_attr + 0.3*torch.randn_like(batch.edge_attr)
#                 edge_p = torch.softmax(edge_p,dim=1)
#                 p_pred = self.discriminator(atom_p,batch.edge_index,edge_p,batch.batch)
#                 fake_p = torch.zeros_like(p_pred)
#                 fake_loss_p = self.adversarial_loss(p_pred,fake_p)
#                 self.log('fake_loss_p',fake_loss_p,prog_bar=True)
#                 if self.global_step<self.warmup_steps:
#                     d_loss = (real_loss + fake_loss_p)/2
#                 else:
#                     d_loss = (2 * real_loss + fake_loss_g + fake_loss_p)/4
#             else:
#                 d_loss = (real_loss + fake_loss_g)/2
#             self.log('real_loss',real_loss,prog_bar=True)
#             self.log('fake_loss_g',fake_loss_g,prog_bar=True)
#             self.log('d_loss',d_loss,prog_bar=True)
#             # stop train discriminator if discriminator is too strong
#             if d_loss<self.d_threshold and real_loss<0.69 and fake_loss_g<0.69 and self.global_step>self.warmup_steps:
#                 return d_loss * 0
#             return d_loss
    
#     def configure_optimizers(self):
#         opt_g = torch.optim.Adam(self.generator.parameters(),lr=0.0002,betas=(0.5,0.999))
#         opt_d = torch.optim.Adam(self.discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
#         return [opt_g,opt_d],[]
    
#     def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
#         checkpoint['g_hparams'] = self.generator.hparams
#         checkpoint['d_hparams'] = self.discriminator.hparams
#         # return super().on_save_checkpoint(checkpoint)

class trainer_wrapper():
    def __init__(self,model:LightningModule,loader,max_epochs:int,\
                 trainer_dir:str,log_name:str,\
                 log_every_n_steps:int=50,\
                 checkpoint_monitor:Optional[str]=None,\
                 save_every_n_steps:int=100,\
                 tqdm_refresh_rate:int=1,**kwargs):
        self.model = model
        self.loader = loader
        self.tf_logger = TensorBoardLogger(trainer_dir,name=log_name)
        if tqdm_refresh_rate==0:
            callbacks = []
        else:
            callbacks = [TQDMProgressBar(refresh_rate=tqdm_refresh_rate)]
        # dirpath = os.path.join(trainer_dir,log_name,'models')
        if checkpoint_monitor is not None:
            checkpoint_callback = ModelCheckpoint(monitor=checkpoint_monitor,\
                                                  save_top_k=2,\
                                                  every_n_train_steps=save_every_n_steps)
        else:
            checkpoint_callback = ModelCheckpoint(every_n_train_steps=save_every_n_steps)
        callbacks.append(checkpoint_callback)
        self.trainer = Trainer(default_root_dir=trainer_dir,\
                               max_epochs=max_epochs,\
                               logger=self.tf_logger,\
                               log_every_n_steps=log_every_n_steps,\
                               callbacks=callbacks,**kwargs)
    def train(self):
        self.trainer.fit(self.model,self.loader)










