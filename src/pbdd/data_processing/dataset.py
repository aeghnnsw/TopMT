import torch
from torch.utils.data import Dataset
from glob import glob
import os
import pickle
import numpy as np

class MVGTorchDataset(Dataset):
    def __init__(self,data_dir:str,box_size:int,include_lig=False):
        # data_dir: path to the directory containing the data
        #           Under data_dir, folders named batch000, batch001, ... are data for PubChem
        #                           folders named ligand_batch000, ... are for ligands in BioLIP
        assert os.path.exists(data_dir), f'{data_dir} does not exist'
        if include_lig:
            files = os.path.join(data_dir,'*','*.pkl')
            self.file_list = glob(files)
        else: 
            files = os.path.join(data_dir,'batch*','*.pkl')
            self.file_list = glob(files)
        self.include_lig = include_lig
        self.box_size = box_size
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        f = open(self.file_list[index],'rb')
        vox_sparse,heatmap_sparse,graph,center = pickle.load(f)
        f.close()
        voxels = torch.zeros(len(vox_sparse),self.box_size,self.box_size,self.box_size)
        i = 0
        for voxel in vox_sparse:
            voxels[i,:,:,:] = voxel.to_dense()
            i += 1
        if self.include_lig:
            return voxels
        indices = torch.tensor(np.ones((200,3))*self.box_size,dtype=torch.long)
        indices_temp = heatmap_sparse._indices().T
        N_atoms,_ = indices_temp.shape
        indices[:N_atoms,:] = indices_temp
        heatmap_indices = indices
        heatmap = heatmap_sparse.to_dense()
        return index,voxels,graph,heatmap_indices,heatmap,center

class DGT_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super().__init__()
        files = os.path.join(data_path,'batch_*','*.pkl')
        self.file_list = glob(files)
        self.N = 50*len(self.file_list)

    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        id1 = int(idx/50)
        id2 = idx%50
        opened=True
        while opened:
            try:
                f = open(self.file_list[id1],'rb')
                opened=False
            except:
                continue
        data_temp = pickle.load(f)
        data = data_temp[id2]
        f.close()
        return data
    
class Type_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,n_samples_per_file=50):
        super().__init__()
        files = os.path.join(data_path,'batch_*','*.pkl')
        self.file_list = glob(files)
        self.n_samples_per_file = n_samples_per_file
        self.N = n_samples_per_file*len(self.file_list)

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        id1 = int(idx/self.n_samples_per_file)
        id2 = idx%self.n_samples_per_file
        opened=False
        while opened==False:
            try:
                f = open(self.file_list[id1],'rb')
                opened=True
            except:
                continue
        data_temp = pickle.load(f)
        data = data_temp[id2]
        f.close()
        return data
    
class Type_Frag_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,n_samples_per_file=10):
        super().__init__()
        files = os.path.join(data_path,'*.pkl')
        self.file_list = glob(files)
        self.n_samples_per_file = n_samples_per_file
        self.N = n_samples_per_file*len(self.file_list)

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        id1 = int(idx/self.n_samples_per_file)
        id2 = idx%self.n_samples_per_file
        opened=False
        while opened==False:
            try:
                f = open(self.file_list[id1],'rb')
                opened=True
            except:
                continue
        data_temp = pickle.load(f)
        data = data_temp[id2]
        f.close()
        return data