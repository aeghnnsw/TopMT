# scripts for dataset creation
import os
import pickle
from tqdm import tqdm
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.utils import to_undirected
from pbdd.data_processing.utils import get_nearest_distance
from pbdd.data_processing.sphere_filling.filling import mol2sphereG


def createIthSphereData(i,data_path,f_sdf,N=50,fill_mode='fcc',radius=1.6,N_max=20,reverse=True,removeHs=True):
    data_path = os.path.join(data_path,f'batch_{i:04d}')
    residue_path = os.path.join(data_path,f'residue')
    os.makedirs(data_path,exist_ok=True)
    os.makedirs(residue_path,exist_ok=True)

    mol_Gs = Chem.SDMolSupplier(f_sdf,removeHs=removeHs)
    d1,d2,d3 = get_nearest_distance(fill_mode=fill_mode,radius=radius)
    n = 0
    batch = 0
    G_list = list()
    for mol in tqdm(mol_Gs):
        if mol is None:
            continue
        d = mol2sphereG(mol,d2,fill_mode=fill_mode,radius=radius,N_max=N_max,reverse=reverse)
        if d is None:
            continue
        G_list.append(d)
        n = n + 1
        if n%N==0:
            f_path = os.path.join(data_path,f'batch{i:04d}_{batch:04d}.pkl')
            f = open(f_path,'wb')
            pickle.dump(G_list,f)
            f.close()
            G_list = list()
            batch = batch + 1
    if len(G_list)!=0:
        f_path = os.path.join(residue_path,f'res_{i:04d}.pkl')
        f = open(f_path,'wb')
        pickle.dump(G_list,f)
        f.close()

def create_ith_type_data(i,data_path,f_sdf,N=100):
    residue_path = os.path.join(data_path,f'residue')
    data_path = os.path.join(data_path,f'batch_{i:04d}')
    os.makedirs(data_path,exist_ok=True)
    os.makedirs(residue_path,exist_ok=True)
    rdkit_mols = Chem.SDMolSupplier(f_sdf,removeHs=True)
    G_list = []
    n = 0
    batch = 0
    for mol in tqdm(rdkit_mols):
        if mol is None:
            continue
        # Kekulize the molecule
        Chem.Kekulize(mol,clearAromaticFlags=True)
        g_temp = mol_to_graph(mol)
        if g_temp is None:
            continue
        G_list.append(g_temp)
        n = n + 1
        if n%N==0:
            f_path = os.path.join(data_path,f'batch{i:04d}_{n//N:04d}.pkl')
            f = open(f_path,'wb')
            pickle.dump(G_list,f)
            f.close()
            G_list = []
            batch = batch + 1
    if len(G_list)!=0:
        f_path = os.path.join(residue_path,f'res_{i:04d}.pkl')
        f = open(f_path,'wb')
        pickle.dump(G_list,f)
        f.close()

def create_frag_type_data(data_dir,f_sdfs,n_mols_per_file=10):
    os.makedirs(data_dir,exist_ok=True)
    G_list = []
    n = 0
    batch = 0
    for f_sdf in f_sdfs:
        rdkit_mols = Chem.SDMolSupplier(f_sdf,removeHs=True)
        for mol in tqdm(rdkit_mols):
            if mol is None:
                continue
            # Kekulize the molecule
            Chem.Kekulize(mol,clearAromaticFlags=True)
            g_temp = mol_to_graph(mol)
            if g_temp is None:
                continue
            G_list.append(g_temp)
            n = n + 1
            if n%n_mols_per_file==0:
                f_path = os.path.join(data_dir,f'batch_{batch:04d}.pkl')
                with open(f_path,'wb') as f:
                    pickle.dump(G_list,f)
                G_list = []
                batch = batch + 1
    return None


def mol_to_graph(mol):
    # convert rdkit mol type to graph type
    # graph has atom as node and bond as edge
    atom_type_dict = {'C':0,'N':1,'O':2,'F':3,'P':4,\
                    'S':5,'Cl':6,'Br':7}
    bondtype_dict = {Chem.rdchem.BondType.SINGLE:0,\
                     Chem.rdchem.BondType.DOUBLE:1,\
                     Chem.rdchem.BondType.TRIPLE:2}
    atom_x = torch.zeros((mol.GetNumAtoms(),8))
    bond_x = torch.zeros((mol.GetNumBonds(),3))
    edge_index = torch.zeros((2,mol.GetNumBonds()),dtype=torch.long)
    for idx,atom_temp in enumerate(mol.GetAtoms()):
        atom_symbol = atom_temp.GetSymbol()
        if atom_symbol in atom_type_dict.keys():
            atom_x[idx,atom_type_dict[atom_symbol]] = 1
        else:
            return None
    for idx,bond_temp in enumerate(mol.GetBonds()):
        id1 = bond_temp.GetBeginAtomIdx()
        id2 = bond_temp.GetEndAtomIdx()
        edge_index[0,idx] = id1
        edge_index[1,idx] = id2
        bond_type = bond_temp.GetBondType()
        if bond_type in bondtype_dict.keys():
            bond_x[idx,bondtype_dict[bond_type]] = 1
        else:
            print(bond_type)
            raise ValueError('Unknown bond type')
            bond_x[idx,bondtype_dict['others']] = 1
    edge_index,bond_x = to_undirected(edge_index,bond_x) # convert to undirected graph
    graph = torch_geometric.data.Data(x=atom_x,edge_index=edge_index,edge_attr=bond_x)
    return graph

        
        

    