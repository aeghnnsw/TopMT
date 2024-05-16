# Models for Deep Graph Translation
# from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch_geometric
# from torch_geometric.nn import aggr
from torch_geometric.nn import MessagePassing,GraphNorm,BatchNorm
# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
# from torch_geometric.nn.models import GAT
# from torch_geometric.utils import subgraph,to_undirected
from torch_geometric.utils import unbatch_edge_index
from typing import Any, Dict, Optional
# from torch.nn import Linear, Sequential, ReLU, Sigmoid,PReLU
import torch.nn.functional as F
from pytorch_lightning import LightningModule
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from pbdd.data_processing.utils import (strip_mol_graph,
                                        draw_graph,
                                        draw_graph_with_type,
                                        perturb_graph,
                                        perturb_graph_type)
from pbdd.models.dgt_models import NECT_layer,NECT_Generator
import os
import numpy as np
from glob import glob
import time

class NECT_Node_Edge_Discriminator(LightningModule):
    # node-level discriminator
    def __init__(self,fn:int,fe:int,num_layers:int,\
                 h1_channels:int,h2_channels:int,\
                 hidden_node_dim:int,hidden_edge_dim:int,\
                 use_pos:bool=False,\
                 node_d:bool=True, edge_d:bool=True,\
                 wgan:bool=False):
        super().__init__()
        self.save_hyperparameters()
        # node level and edge level at least one is True
        assert node_d or edge_d,'node_d and edge_d cannot be both False'
        self.num_layers = num_layers
        self.use_pos = use_pos
        self.node_d = node_d
        self.edge_d = edge_d
        self.wgan = wgan
        self.layers = torch.nn.ModuleList()
        
        # wgan does not use BatchNorm
        if not wgan:
            self.graph_norm_layer = torch.nn.ModuleList()
            self.edge_batch_norm_layer = torch.nn.ModuleList()
        fn1 = fn
        fe1 = fe
        for i in range(num_layers):
            self.layers.append(NECT_layer(fn1,fe1,h1_channels,h2_channels,\
                                          hidden_node_dim,hidden_edge_dim,\
                                          use_pos=use_pos))
            fn1 = hidden_node_dim
            fe1 = hidden_edge_dim
            if not wgan:
                self.graph_norm_layer.append(GraphNorm(hidden_node_dim))
                self.edge_batch_norm_layer.append(BatchNorm(hidden_edge_dim))
        if self.node_d:
            self.node_mlp = torch.nn.Sequential(torch.nn.Linear(hidden_node_dim,hidden_node_dim//2),\
                                                torch.nn.PReLU(),\
                                                torch.nn.Linear(hidden_node_dim//2,1))
        if self.edge_d:
            self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(hidden_edge_dim,hidden_edge_dim//2),\
                                                torch.nn.PReLU(),\
                                                torch.nn.Linear(hidden_edge_dim//2,1))
    
    def forward(self,x,edge_index,edge_attr,batch,pos:Optional[torch.Tensor]=None):
        if self.wgan:
            for i in range(self.num_layers):
                x,edge_attr = self.layers[i](x,edge_index,edge_attr,pos=pos)
        else:
            for i in range(self.num_layers):
                x,edge_attr = self.layers[i](x,edge_index,edge_attr,pos=pos)
                x = self.graph_norm_layer[i](x,batch)
                edge_attr = self.edge_batch_norm_layer[i](edge_attr)
        if self.node_d:
            x = self.node_mlp(x)
        if self.edge_d:
            edge_attr = self.edge_mlp(edge_attr)
        # x1 = global_add_pool(x,batch)
        # x2 = global_mean_pool(x,batch)
        # x3 = global_max_pool(x,batch)
        # x = torch.cat((x1,x2,x3),dim=1)
        # x = self.mlp(x)
        return x,edge_attr

class mol_assign_GAN(LightningModule):
    def __init__(self,generator:LightningModule,discriminator:LightningModule,\
                 d_threshold:float=0.1,perturb:bool=False,warmup_steps:int=1000,\
                 ele_loss:bool=False):
        super().__init__()
        self.save_hyperparameters(ignore=['generator','discriminator'])
        self.generator = generator
        self.discriminator = discriminator
        assert hasattr(self.discriminator,'wgan'),'discriminator must has attribute wgan'
        assert hasattr(self.discriminator,'node_d'),'discriminator must has attribute node_d'
        assert hasattr(self.discriminator,'edge_d'),'discriminator must has attribute edge_d'
        self.cpu_device = torch.device('cpu')
        # self.d_threshold = d_threshold
        self.perturb = perturb
        self.warmup_steps = warmup_steps
        self.ele_loss = ele_loss
        self.node_d = self.discriminator.node_d
        self.edge_d = self.discriminator.edge_d
        self.wgan = self.discriminator.wgan

    
    def forward(self,z,edge_index,edge_attr,batch):
        x,edge_out = self.generator(z,edge_index,edge_attr,batch)
        # perform softmax on node and edge
        x = torch.softmax(x,dim=1)
        edge_out = torch.softmax(edge_out,dim=1)
        return x,edge_out
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy_with_logits(y_hat,y)
    
    def element_loss(self,y_hat,y):
        return F.mse_loss(y_hat,y)
    
    def calc_gradient_penalty(self,edge_index,real_atoms,fake_atoms,\
                              real_edges,fake_edges,ptr,batch):
        # calculate gradient penalty
        # real_data and fake_data are graph batch data
        # for p in self.discriminator.parameters():
            # p.requires_grad = True
        alpha_atom = torch.zeros(real_atoms.shape[0],1).type_as(real_atoms)
        alpha_edge = torch.zeros(real_edges.shape[0],1).type_as(real_edges)
        num_graphs = len(ptr)-1
        for i in range(num_graphs):
            ptr1 = ptr[i]
            ptr2 = ptr[i+1]
            alpha_atom[ptr1:ptr2] = torch.rand(1)
        edge_list = unbatch_edge_index(edge_index,batch)
        num_edges = [len(edge[0]) for edge in edge_list]
        ptr1 = 0
        for i in range(num_graphs):
            ptr2 = ptr1 + num_edges[i]
            alpha_edge[ptr1:ptr2] = torch.rand(num_edges[i],1)
            ptr1 = ptr2
        inter_atoms = alpha_atom * real_atoms + ((1 - alpha_atom) * fake_atoms)
        inter_edges = alpha_edge * real_edges + ((1 - alpha_edge) * fake_edges)
        inter_atoms.requires_grad = True
        inter_edges.requires_grad = True
        node_pred,edge_pred = self.discriminator(inter_atoms,edge_index,inter_edges,batch)
        # print(node_pred)
        # print(edge_pred)
        

        gradients = torch.autograd.grad(outputs=[node_pred,edge_pred],\
                                        inputs=[inter_atoms,inter_edges],\
                                        grad_outputs=[torch.ones_like(node_pred),\
                                                      torch.ones_like(edge_pred)],\
                                        create_graph=True,allow_unused=True)
        # print(gradients)
        # gradients = torch.autograd.grad(outputs=node_pred,\
        #                                 inputs=(inter_atoms,inter_edges),\
        #                                 grad_outputs=torch.ones_like(node_pred),\
        #                                 create_graph=True,retain_graph=True,only_inputs=True)


        node_grad = gradients[0].view(gradients[0].shape[0],-1)
        edge_grad = gradients[1].view(gradients[1].shape[0],-1)
        node_grad_norm = node_grad.norm(2,dim=1)
        edge_grad_norm = edge_grad.norm(2,dim=1)
        node_penalty = ((node_grad_norm-1)**2).mean()
        edge_penalty = ((edge_grad_norm-1)**2).mean()
        return node_penalty + edge_penalty

    def training_step(self,batch,batch_idx,optimizer_idx):
        # sample noise
        z = torch.randn(batch.x.shape[0],1).type_as(batch.x)
        edge_z = torch.randn(batch.edge_attr.shape[0],1).type_as(batch.x)
        # generate node type and edge type
        atom_g,edge_g = self(z,batch.edge_index,edge_z,batch.batch)
        # train generator
        if optimizer_idx == 0:
            atom_g_pred,edge_g_pred = self.discriminator(atom_g,batch.edge_index,edge_g,batch.batch)
            atom_valid = torch.ones_like(atom_g_pred)
            edge_valid = torch.ones_like(edge_g_pred)
            if self.wgan:
                g_atom_loss = -torch.mean(atom_g_pred)
                g_edge_loss = -torch.mean(edge_g_pred)
                g_loss = g_atom_loss + g_edge_loss
                self.log('g_atom_w',-g_atom_loss,prog_bar=True)
                self.log('g_edge_w',-g_edge_loss,prog_bar=True)
                self.log('g_w',-g_loss,prog_bar=True)
            else:
                g_atom_loss = self.adversarial_loss(atom_g_pred,atom_valid)
                g_edge_loss = self.adversarial_loss(edge_g_pred,edge_valid)
                g_loss = g_atom_loss + g_edge_loss
                self.log('g_atom_loss',g_atom_loss,prog_bar=True)
                self.log('g_edge_loss',g_edge_loss,prog_bar=True)
                self.log('g_loss',g_loss,prog_bar=True)

            # log images, generate two images to check mode collapse
            if self.global_step % 200 == 0:
                sample_data = batch.to_data_list()[0]
                for i in range(2):
                    batch_temp = torch.zeros(sample_data.x.shape[0]).type_as(batch.batch)
                    z_temp = torch.randn(sample_data.x.shape[0],1).type_as(sample_data.x)
                    edge_z_temp = torch.randn(sample_data.edge_attr.shape[0],1).type_as(sample_data.x)
                    atom_g_temp,edge_g_temp = self(z_temp,sample_data.edge_index,\
                                                   edge_z_temp,batch_temp)
                    g_temp = torch_geometric.data.Data(x=atom_g_temp.to(self.cpu_device),\
                                                       edge_index=sample_data.edge_index,\
                                                       edge_attr=edge_g_temp.to(self.cpu_device))
                    np_fig = draw_graph_with_type(g_temp,show=False,return_np=True)
                    self.logger.experiment.add_image(f'generated_graph{i}',np_fig,self.global_step//2)
            if self.global_step<self.warmup_steps:
                g_loss = g_loss * 0
            else:
                # element_loss
                if self.ele_loss:
                    real_r_mean = torch.mean(batch.x,dim=0)
                    atom_g_mean = torch.mean(torch.softmax(5*atom_g,dim=1),dim=0)
                    ele_loss = self.element_loss(atom_g_mean,real_r_mean)
                    self.log('ele_loss',ele_loss,prog_bar=True)
                    g_loss = g_loss + 0.5*ele_loss
            return g_loss
        # train discriminator
        if optimizer_idx == 1:
            # perturb atom and bond type
            atom_r = 3*batch.x + 0.1*torch.randn_like(batch.x)
            atom_r = torch.softmax(atom_r,dim=1)
            edge_r = 2*batch.edge_attr + 0.2*torch.randn_like(batch.edge_attr)
            edge_r = torch.softmax(edge_r,dim=1)
            atom_r_pred,edge_r_pred = self.discriminator(atom_r,batch.edge_index,\
                                                         edge_r,batch.batch)
            atom_g_pred,edge_g_pred = self.discriminator(atom_g.detach(),batch.edge_index,\
                                                         edge_g.detach(),batch.batch)
            if self.wgan:
                atom_real_loss = -torch.mean(atom_r_pred)
                edge_real_loss = -torch.mean(edge_r_pred)
                atom_fake_loss_g = torch.mean(atom_g_pred)
                edge_fake_loss_g = torch.mean(edge_g_pred)
                real_loss = atom_real_loss + edge_real_loss
                fake_loss_g = atom_fake_loss_g + edge_fake_loss_g
                self.log('atom_real_w',-atom_real_loss,prog_bar=True)
                self.log('edge_real_w',-edge_real_loss,prog_bar=True)
                self.log('real_w',-real_loss,prog_bar=True)
            else:
                atom_valid = torch.ones_like(atom_r_pred)
                edge_valid = torch.ones_like(edge_r_pred)
                atom_fake_g = torch.zeros_like(atom_g_pred)
                edge_fake_g = torch.zeros_like(edge_g_pred)
                atom_real_loss = self.adversarial_loss(atom_r_pred,atom_valid)
                edge_real_loss = self.adversarial_loss(edge_r_pred,edge_valid)
                real_loss = atom_real_loss + edge_real_loss
                atom_fake_loss_g = self.adversarial_loss(atom_g_pred,atom_fake_g)
                edge_fake_loss_g = self.adversarial_loss(edge_g_pred,edge_fake_g)
                fake_loss_g = atom_fake_loss_g + edge_fake_loss_g
                self.log('atom_real_loss',atom_real_loss,prog_bar=True)
                self.log('edge_real_loss',edge_real_loss,prog_bar=True)
                self.log('atom_fake_loss_g',atom_fake_loss_g,prog_bar=True)
                self.log('edge_fake_loss_g',edge_fake_loss_g,prog_bar=True)
                self.log('real_loss',real_loss,prog_bar=True)
                self.log('fake_loss_g',fake_loss_g,prog_bar=True)
            if self.perturb:
                alpha = np.random.uniform(0.2,0.9)
                atom_p,edge_p = perturb_graph_type(batch.edge_index,batch.x,batch.edge_attr,alpha)
                atom_p = 3*batch.x + 0.2*torch.randn_like(batch.x)
                atom_p = torch.softmax(atom_p,dim=1)
                edge_p = 2*batch.edge_attr + 0.3*torch.randn_like(batch.edge_attr)
                edge_p = torch.softmax(edge_p,dim=1)
                atom_p_pred,edge_p_pred = self.discriminator(atom_p,batch.edge_index,edge_p,batch.batch)
                if self.wgan:
                    atom_fake_loss_p = torch.mean(atom_p_pred)
                    edge_fake_loss_p = torch.mean(edge_p_pred)
                    fake_loss_p = atom_fake_loss_p + edge_fake_loss_p
                    self.log('p_atom_w',atom_fake_loss_p,prog_bar=True)
                    self.log('p_edge_w',edge_fake_loss_p,prog_bar=True)
                    self.log('p_w',fake_loss_p,prog_bar=True)
                    self.log('w_dist_p',real_loss+fake_loss_g,prog_bar=True)
                else:
                    atom_fake_p = torch.zeros_like(atom_p_pred)
                    edge_fake_p = torch.zeros_like(edge_p_pred)
                    atom_fake_loss_p = self.adversarial_loss(atom_p_pred,atom_fake_p)
                    edge_fake_loss_p = self.adversarial_loss(edge_p_pred,edge_fake_p)
                    fake_loss_p = atom_fake_loss_p + edge_fake_loss_p
                    self.log('atom_fake_loss_p',atom_fake_loss_p,prog_bar=True)
                    self.log('edge_fake_loss_p',edge_fake_loss_p,prog_bar=True)
                    self.log('fake_loss_p',fake_loss_p,prog_bar=True)
                if self.global_step<self.warmup_steps:
                    d_loss = (real_loss + fake_loss_p)/2
                else:
                    d_loss = (2 * real_loss + fake_loss_g + fake_loss_p)/4
            else:
                d_loss = (real_loss + fake_loss_g)/2
            # gradient penalty
            if self.wgan:
                penalty = self.calc_gradient_penalty(batch.edge_index,atom_r,atom_g.detach(),\
                                                     edge_r,edge_g.detach(),batch.ptr,batch.batch)
                d_loss = d_loss + 10*penalty
                self.log('grad_penalty',penalty,prog_bar=True)
                self.log('w_dist_g',real_loss+fake_loss_g,prog_bar=True)
            self.log('d_loss',d_loss,prog_bar=True)
            # stop train discriminator if discriminator is too strong
            # if not self.wgan:
                # if d_loss<self.d_threshold and real_loss<0.69 and fake_loss_g<0.69 and self.global_step>self.warmup_steps:
                    # return d_loss * 0
            return d_loss
    
    def configure_optimizers(self):
        if self.wgan:
            opt_g = torch.optim.RMSprop(self.generator.parameters(),lr=0.00005)
            opt_d = torch.optim.RMSprop(self.discriminator.parameters(),lr=0.00005)
        else:
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
        discriminator = NECT_Node_Edge_Discriminator(**d_hparams)
        model = mol_assign_GAN(generator,discriminator)
        model.load_state_dict(params['state_dict'])
        return model

