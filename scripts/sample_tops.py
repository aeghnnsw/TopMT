#!/usr/bin/env python3
import argparse
import os
import pickle
from time import time

import numpy as np
# from glob import glob
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from pbdd.data_processing.sphere_filling.filling import (get_grids_from_mol2,
                                                         get_grids_from_pdb)
from pbdd.models.dgt_models import DGT_GAN
# from pbdd.models.ma_models import mol_assign_GAN
from pbdd.post_processing.sample import sample_graphs, sample_tops
from pbdd.post_processing.scoring import topology_freq_filter
from pbdd.post_processing.utils import convert_rdkit_mol, to_directed


# When using pocket mode, especially larger pockets,
# it is recommended to use smaller sample_repeats and larger sample_tol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='scaffold_hopping',\
                        choices=['scaffold_hopping','pocket'],\
                        help='mode of sampling tops')
    parser.add_argument('--ori_lig_file',type=str, default=None,\
                        help='path to the original ligand mol2 or pdb file')
    parser.add_argument('--pocket_file',type=str, default=None,\
                        help='path to the pocket pdb file')
    parser.add_argument('--n_max_spheres',type=int,default=80,\
                        help='maximum number of spheres for stacking and sampling tops')
    parser.add_argument('--min_atoms',type=int,default=None,\
                        help='minimum number of atoms for sampling tops')
    parser.add_argument('--max_atoms',type=int,default=None,\
                        help='maximum number of atoms for sampling tops')
    parser.add_argument('--rot_bonds',type=int,default=None,\
                        help='number of rotatable bonds for sampling tops')
    parser.add_argument('--nrot_upper',type=int,default=2,\
                        help='number of rotatable bonds upper bound for sampling tops')
    parser.add_argument('--nrot_lower',type=int,default=2,\
                        help='number of rotatable bonds lower bound for sampling tops')
    parser.add_argument('--natoms_lower',type=int,default=2,\
                        help='number of atoms lower bound for sampling tops')
    parser.add_argument('--natoms_upper',type=int,default=2,\
                        help='number of atoms upper bound for sampling tops')
    parser.add_argument('--search_gs',type=int,default=128,\
                        help='minimum number of graphs entered into the search process')
    parser.add_argument('--search_steps',type=int,default=200,\
                        help='number of search steps for sampling tops')
    parser.add_argument('--search_repeats',type=int,default=5,\
                        help='number of search repeats for sampling tops')
    # parser.add_argument('--pocket_pdbqt',type=str,\
                        # help='path to the pocket pdbqt file')
    parser.add_argument('--wkdir',type=str,\
                        help='path to the working directory')

    # sample top parameters
    parser.add_argument('--grid_boundary',type=float,default=0.5,\
                        help='boundary of the grid')
    parser.add_argument('--DGT_ckpt_path',type=str,\
                        help='path to the DGT GAN checkpoint')
    parser.add_argument('--n_tops',type=int,default=1024,\
                        help='number of topologies to sample')
    parser.add_argument('--sample_repeats',type=int,default=100,\
                        help='number of repeats for sampling one stacking')
    parser.add_argument('--sample_tol',type=int,default=200,\
                        help='number of sample cycles before relax the constraints')
    parser.add_argument('--batch_size',type=int,default=48,\
                        help='batch size for sampling tops')
    parser.add_argument('--threshold',type=float,default=0.35,\
                        help='threshold for sampling tops')
    parser.add_argument('--top_freq_lib',type=str,default=None,\
                        help='path to the topology frequency library')
    # parser.add_argument('--cuda_id',type=int,default=0,\
                        # help='cuda id for inference')
    # assgin mol parameters
    # parser.add_argument('--mol_assign_ckpt_path',type=str,\
    #                     help='path to the mol assign GAN checkpoint')
    # parser.add_argument('--n_assigns',type=int,default=100,\
    #                     help='number of assignments for each top to sample')
    # parser.add_argument('--batch_size',type=int,default=10000,\
    #                     help='batch size for saving assigned mols')

    # filtering parameters
    # parser.add_argument('--SAS_quentile',type=float,default=0.9,\
    #                     help='SAS quentile threshold for filtering, smaller is better')
    # parser.add_argument('--QED_quentile',type=float,default=0.9,\
    #                     help='QED quentile threshold for filtering, larger is better')
    # parser.add_argument('--vina_threshold',type=float,default=-8.0,\
    #                     help='vina score threshold for saving poses')

    t1 = time()
    args = parser.parse_args()

    if args.mode=='scaffold_hopping':
        assert args.ori_lig_file is not None, 'ori_lig_file must be provided in scaffold hopping mode'
        pocket_file = args.ori_lig_file
    elif args.mode=='pocket':
        assert args.pocket_file is not None, 'pocket_file must be provided in pocket mode'
        if args.ori_lig_file is None:
            assert args.min_atoms is not None, 'min_atoms must be provided if ori_lig_file is not provided'
            assert args.max_atoms is not None, 'max_atoms must be provided if ori_lig_file is not provided'
            assert args.rot_bonds is not None, 'rot_bonds must be provided if ori_lig_file is not provided'
        pocket_file = args.pocket_file

    wkdir = args.wkdir

    grid_boundary = args.grid_boundary
    # pocket_pdbqt = args.pocket_pdbqt
    DGT_GAN_ckpt_path = args.DGT_ckpt_path
    n_tops = args.n_tops
    sample_repeats = args.sample_repeats
    top_freq_lib = args.top_freq_lib
    print('top_freq_lib',top_freq_lib)

    # mol_assign_ckpt_path = args.mol_assign_ckpt_path
    # n_assigns = args.n_assigns
    # batch_size = args.batch_size
    # mol_assign_dir = os.path.join(wkdir,'mols_assign')
    # os.makedirs(mol_assign_dir,exist_ok=True)
    # mol_assign_path = os.path.join(mol_assign_dir,'mols_assign.pkl')

    # vina_threshold = args.vina_threshold

    os.makedirs(wkdir,exist_ok=True)
    top_path = os.path.join(wkdir,'tops.pkl')


    # ### Get pocket info
    pocket_file_type = pocket_file.split('.')[-1]
    # if pocket_file_type=='pdb':
    #     pocket_mol = Chem.MolFromPDBFile(pocket_file, removeHs=True)
    # elif pocket_file_type=='mol2':
    #     pocket_mol = Chem.MolFromMol2File(pocket_file, removeHs=True)
    # else:
    #     raise ValueError(f'unsupported ligand file type: {pocket_file_type}')
    
    if args.ori_lig_file is not None:
        ori_lig_file_type = args.ori_lig_file.split('.')[-1]
        if ori_lig_file_type=='pdb':
            ori_mol = Chem.MolFromPDBFile(args.ori_lig_file, removeHs=True)
        elif ori_lig_file_type=='mol2':
            ori_mol = Chem.MolFromMol2File(args.ori_lig_file, removeHs=True)
        num_atoms = ori_mol.GetNumAtoms()
        num_rot = Descriptors.NumRotatableBonds(ori_mol)
        pos = ori_mol.GetConformer().GetPositions()
        pos_min = np.min(pos,axis=0)
        pos_max = np.max(pos,axis=0)
        lig_pos_center = (pos_min+pos_max)/2
        print('Original ligand Info:')
        print('num_atoms: ',num_atoms)
        print('num_rotatble bonds: ',num_rot)
        print('pos_center: ',lig_pos_center)
        n_atom_min = num_atoms-args.natoms_lower
        n_atom_max = num_atoms+args.natoms_upper
    else:
        n_atom_min = args.min_atoms
        n_atom_max = args.max_atoms
    if args.rot_bonds is not None:
        num_rot = args.rot_bonds
    n_rot_min = max(num_rot-args.nrot_lower,0)
    n_rot_max = num_rot+args.nrot_upper


    ### Sample topologies

    gpu_availabel = torch.cuda.is_available()
    if gpu_availabel:
        device = torch.device(f'cuda:0')
    else:
        device = torch.device('cpu')
    # print(f'sample on {device}')

    trained_gan = DGT_GAN.load_GAN_from_checkpoint(DGT_GAN_ckpt_path).to(device)
    if top_freq_lib is not None:
        with open(top_freq_lib,'rb') as f:
            topology_freq_dict = pickle.load(f)
    else:
        print('no topology frequency library provided, skip topology filtering')
        topology_freq_dict = None

    top_list = []
    smi_list = []
    sample_counts = 0
    while len(top_list)<n_tops:
        gs = []
        n_gs = 0
        while len(gs)<args.search_gs:
            if pocket_file_type=='pdb':
                grids = get_grids_from_pdb(pocket_file,boundary=grid_boundary,n_max_nodes=-1,\
                                        n_max_spheres=args.n_max_spheres)
            elif pocket_file_type=='mol2':
                grids = get_grids_from_mol2(pocket_file,boundary=grid_boundary)
            # print(grids)
            edge_index = grids.edge_index
            pos = grids.pos
            pos_center = grids.pos_center
            rot_angles = grids.rot_angles
            restore_dict ={'pos_center':pos_center,'rot_angles':rot_angles}
            for i in range(sample_repeats):
                gs_temp = sample_graphs(trained_gan,edge_index,pos,n_preds=args.batch_size,\
                                        min_atoms=n_atom_min,threshold=args.threshold,\
                                        device=device,**restore_dict)
                gs.extend(gs_temp)
            if len(gs)>n_gs:
                n_gs = len(gs)
                print(f'number of graphs sampled: {n_gs}')
            # print('number of graphs sampled: ',len(gs))
        print(f'sample counts: {sample_counts} candidate graphs: {len(gs)}')
        mol_temps,results = sample_tops(gs,n_atom_min,n_atom_max,n_rot_min,n_rot_max,\
                                        args.search_repeats,args.search_steps)
        if len(mol_temps)>0:
            for mol in mol_temps:
                rdkit_top_temp = convert_rdkit_mol(to_directed(mol.edge_index))[0]
                smi = Chem.MolToSmiles(rdkit_top_temp)
                if smi in smi_list:
                    continue
                if topology_freq_dict is not None:
                    if topology_freq_filter(mol,topology_freq_dict)==False:
                        continue
                smi_list.append(smi)
                top_list.append(mol)
        t_temp = time()
        print(f'current valid graphs: {len(top_list)}, time: {t_temp-t1}')
        sample_counts +=1
        if sample_counts%args.sample_tol==0:
            n_rot_min -= 1
            if n_rot_max<12:
                n_rot_max += 1

    # save all top
    with open(top_path,'wb') as f:
        pickle.dump(top_list,f)

    t2 = time()
    print(f'sample top time: {t2-t1}')

if __name__=='__main__':
    main()

# with open(top_path,'rb') as f:
#     top_list = pickle.load(f)
# print(f'number of tops: {len(top_list)}')

