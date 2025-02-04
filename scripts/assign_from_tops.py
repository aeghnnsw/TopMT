#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import pickle
from time import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from pbdd.models.ma_models import mol_assign_GAN
from pbdd.post_processing.sample import assign_mols
from pbdd.post_processing.scoring import vina_score_with_convert
from pbdd.post_processing.utils import convert_rdkit_pdbqt_str


def convert_sdf_to_pdbqt(assign_sdf_path,assign_pdbqt_path):
    if os.path.exists(assign_pdbqt_path):
        return None
    else:
        os.system(f'obabel -isdf {assign_sdf_path} -opdbqt -O {assign_pdbqt_path} -p 7.4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_lig_file',type=str,\
                        help='path to the original ligand mol2 file')
    parser.add_argument('--pocket_pdbqt',type=str,\
                        help='path to the pocket pdbqt file')
    parser.add_argument('--wkdir',type=str,\
                        help='path to the working directory')

    parser.add_argument('--top_save_path',type=str,default=None,\
                        help='path to save the sampled tops')
    # assgin mol parameters
    parser.add_argument('--assign_method',type=str,default='GAN',choices=['GAN','Matching'],\
                        help='method to assign mols, choose from GAN and Matching')
    parser.add_argument('--mol_assign_ckpt_path',type=str,default=None,\
                        help='path to the mol assign GAN checkpoint')
    parser.add_argument('--n_assigns',type=int,default=100,\
                        help='number of assignments for each top to sample')
    parser.add_argument('--load',action='store_true',\
                        help='whether to load the assigned mols from pickle, do this is'+\
                        'you already have assigned mols')
    parser.add_argument('--save_batch_size',type=int,default=10000,\
                        help='batch size for saving assigned mols')
    parser.add_argument('--n_subjobs',type=int,default=1,\
                        help='number of subjobs for assigning mols')
    parser.add_argument('--subjob_rank',type=int,default=0,\
                        help='rank of the current subjob')
    parser.add_argument('--n_process',type=int,default=64,\
                        help='number of processes for multiprocessing, default is 0 to decide automatically')

    # filtering parameters
    parser.add_argument('--vina_threshold',type=float,default=-8.0,\
                        help='vina score threshold for saving poses')
    parser.add_argument('--qed_threshold',type=float,default=0.5,\
                        help='QED threshold for filtering, value between 0 and 1')
    # parser.add_argument('--sas_threshold',type=float,default=8.0,\
    #                     help='SAS threshold for filtering')
    parser.add_argument('--box_length',type=float,default=30.0,\
                        help='box length for vina docking')
    parser.add_argument('--max_step',type=int,default=100,\
                        help='max steps for vina optimization')
    parser.add_argument('--write_pose',action='store_true',\
                        help='whether to write the pose')

    t0 = time()
    args = parser.parse_args()


    ori_lig_file = args.ori_lig_file
    wkdir = args.wkdir

    pocket_pdbqt = args.pocket_pdbqt
    # DGT_GAN_ckpt_path = args.DGT_ckpt_path
    # n_tops = args.n_tops
    # sample_repeats = args.sample_repeats
    # top_freq_lib = args.top_freq_lib
    assign_method = args.assign_method
    top_save_path = args.top_save_path
    if top_save_path is None:
        if assign_method=='GAN':
            top_save_path = os.path.join(wkdir,'tops.pkl')
        else:
            raise NotImplementedError('Matching method is not implemented yet')
    assert os.path.exists(top_save_path),f'{top_save_path} does not exist'
    assert args.qed_threshold>=0 and args.qed_threshold<=1,\
        f'qed_threshold should be between 0 and 1, got {args.qed_threshold}'

    if assign_method=='GAN':
        mol_assign_ckpt_path = args.mol_assign_ckpt_path
        assert os.path.exists(mol_assign_ckpt_path),f'{mol_assign_ckpt_path} does not exist'
        n_assigns = args.n_assigns

    else:
        raise NotImplementedError('Matching method is not implemented yet')

    save_batch_size = args.save_batch_size
    n_subjobs = args.n_subjobs
    subjob_rank = args.subjob_rank

    print('n_subjobs: ',n_subjobs)
    print('subjob_rank: ',subjob_rank)

    mol_assign_dir = os.path.join(wkdir,f'mols_assign_{subjob_rank+1}_of_{n_subjobs}')
    os.makedirs(mol_assign_dir,exist_ok=True)
    mol_assign_path = os.path.join(mol_assign_dir,'mols_assign.pkl')
    vina_threshold = args.vina_threshold

    # os.makedirs(wkdir,exist_ok=True)

    # ### Get original ligand info
    assert os.path.exists(ori_lig_file),f'{ori_lig_file} does not exist'
    lig_file_type = ori_lig_file.split('.')[-1]
    if lig_file_type=='pdb':
        ori_mol = Chem.MolFromPDBFile(ori_lig_file, removeHs=False)
    elif lig_file_type=='mol2':
        ori_mol = Chem.MolFromMol2File(ori_lig_file, removeHs=False)
    else:
        raise ValueError(f'unsupported ligand file type: {lig_file_type}')
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


    if args.load == False or not os.path.exists(mol_assign_path):
        with open(top_save_path,'rb') as f:
            top_list = pickle.load(f)

        # get subjob top_list
        n_tops = len(top_list)
        print('Number of tops:', n_tops)

        # Calculate the basic number of tops per subjob
        base_batch_size = n_tops // n_subjobs
        # Calculate the number of subjobs that will receive an extra top
        extra_tops = n_tops % n_subjobs

        top_list_subjobs = []
        current_index = 0

        for rank_temp in range(n_subjobs):
            if rank_temp < extra_tops:
                # This subjob gets an extra top
                batch_size = base_batch_size + 1
            else:
                batch_size = base_batch_size
            
            subjob_tops = top_list[current_index:current_index + batch_size]
            top_list_subjobs.append(subjob_tops)
            current_index += batch_size
        
        top_list = top_list_subjobs[subjob_rank]
        print('number of tops in subjob: ',len(top_list))

        mol_assign_trained_model = mol_assign_GAN.load_GAN_from_checkpoint(mol_assign_ckpt_path)
        assigned_mols = assign_mols(top_list,mol_assign_trained_model,\
                                        n_samples=n_assigns,qed_threshold=args.qed_threshold)

        if len(assigned_mols) == 0:
            print('no mols assigned, please check the assign method and parameters')
            return None
        # save assigned mols as pickle
        with open(mol_assign_path,'wb') as f:
            pickle.dump(assigned_mols,f)
    else:
        #load assigned mols from pickle
        with open(mol_assign_path,'rb') as f:
            assigned_mols = pickle.load(f)
    # calculate SAS and QED and filter based on quentile
    # n_mols = len(assigned_mols)
    # SAS_list = []
    # QED_list = []
    # for i,mol in enumerate(assigned_mols):
    #     sas_temp = calc_sas(mol)
    #     qed_temp = calc_qed(mol)
    #     SAS_list.append([i,sas_temp])
    #     QED_list.append([i,qed_temp])
    #     mol.SetProp('qed',f'{qed_temp:.3f}')
    #     mol.SetProp('sas',f'{sas_temp:.3f}')
    # SAS_list.sort(key=lambda x:x[1])
    # QED_list.sort(key=lambda x:x[1],reverse=True)
    # SAS_keep_n = int(n_mols*(1-args.SAS_quentile))
    # QED_keep_n = int(n_mols*(1-args.QED_quentile))
    # SAS_keep_index = [i for i,_ in SAS_list[:SAS_keep_n]]
    # QED_keep_index = [i for i,_ in QED_list[:QED_keep_n]]
    # keep_index = list(set(SAS_keep_index).intersection(set(QED_keep_index)))

    print(f'number of assigned mols: {len(assigned_mols)}')
    batch_id = 0
    assign_sdf_path = os.path.join(mol_assign_dir,f'mol_batch_{batch_id:03d}.sdf')
    w = Chem.SDWriter(assign_sdf_path)
    pos_dir = os.path.join(mol_assign_dir,'pos')
    os.makedirs(pos_dir,exist_ok=True)
    mol_dict = {}
    for i,mol in enumerate(assigned_mols):
        if i%save_batch_size==0 and i>0:
            batch_id += 1
            assign_sdf_path = os.path.join(mol_assign_dir,\
                                        f'mol_batch_{batch_id:03d}.sdf')
            w = Chem.SDWriter(assign_sdf_path)
        name_temp = f'mol_{subjob_rank}_{batch_id:03d}_{i%save_batch_size:04d}'
        mol.SetProp('_Name',name_temp)
        w.write(mol)
        mol_dict[name_temp] = mol
       
    w.close()
    t1 = time()

    # convert sdf to pdbqt, multiprocess
 
    # convert_task_list = []
    # num_process = os.cpu_count()//2
    # num_process = min(num_process,batch_id+1)
    # for i in range(batch_id+1):
    #     assign_sdf_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.sdf')
    #     assign_pdbqt_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.pdbqt')
    #     convert_task_list.append((assign_sdf_path,assign_pdbqt_path))
    # with multiprocessing.get_context('fork').Pool(num_process) as pool:
    #     pool.starmap(convert_sdf_to_pdbqt,convert_task_list)

    # vina scoring and save poses with score<threshold
    
 

    score_list = []

    for i in range(batch_id+1):
        t21 = time()
        print('score batch ',i)
        temp_sdf_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.sdf')
        # temp_pdbqt_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.pdbqt')
        # read pdbqt and split to string list
        sdf_suppl = Chem.SDMolSupplier(temp_sdf_path)
        # pdbqt_str_list = []
        # for mol in tqdm(sdf_suppl):
            # pdbqt_str = convert_rdkit_pdbqt_str(mol,add_h=True)
            # pdbqt_str_list.append(pdbqt_str)
        # with open(temp_pdbqt_path,'r') as f:
            # lines = f.readlines()
            # mol_temp = ''
            # mol_str_list = []
            # for line in lines:
            #     if line.startswith('MODEL'):
            #         continue
            #     if line.startswith('ENDMDL'):
            #         mol_str_list.append(mol_temp)
            #         mol_temp = ''
            #         continue
            #     mol_temp += line
        # vina scoring
        # n_mols = len(pdbqt_str_list)
        vina_task_list = []
        # num_process = os.cpu_count()-2
        for j,mol_temp in enumerate(sdf_suppl):
            pose_name = os.path.join(pos_dir,f'mol_{i:03d}_{j:05d}')
            vina_task_list.append((mol_temp,pocket_pdbqt,lig_pos_center,\
                                   pose_name,vina_threshold,args.box_length,args.write_pose,\
                                   args.max_step))
        with multiprocessing.get_context('forkserver').Pool(args.n_process) as pool:
            score_temp = pool.starmap(vina_score_with_convert,vina_task_list)
        score_list.append(score_temp)
        t22 = time()
        print(f'batch {i} scoring time: {t22-t21}')

    # # save scores and ranking

    # collect all scores and corresponding poses
    # print(score_list)
    print(len(score_list))
    collect_scores = []
    for batch_i,scores in enumerate(score_list):
        for mol_j,score in enumerate(scores):
            collect_scores.append([subjob_rank,batch_i,mol_j,score[0]])
    # sort by score
    collect_scores.sort(key=lambda x:x[3])

    # save scores
    if assign_method == 'GAN':
        score_dir = os.path.join(wkdir,'scores')
    elif assign_method == 'Matching':
        score_dir = os.path.join(wkdir,'match_scores')
    os.makedirs(score_dir,exist_ok=True)
    score_path = os.path.join(score_dir,f'scores_{subjob_rank}.pkl')
    with open(score_path,'wb') as f:
        pickle.dump(collect_scores,f)


    # read top poses and write to sdf (only score<threshold)
    top_mols = []
    mols_smi = []
    n_poses = 0
    # print(collect_scores)
    for _,batch_i,mol_j,score in collect_scores:
        if score>=vina_threshold:
            break
        mol_name = f'mol_{subjob_rank}_{batch_i:03d}_{mol_j:04d}'
        mol_temp = mol_dict[mol_name]
        smi_temp = Chem.MolToSmiles(mol_temp)
        if smi_temp in mols_smi:
            continue
        mol_temp.SetProp('vina_score',str(score))
        top_mols.append(mol_temp)
        n_poses += 1

    # write top poses sdf
    if assign_method == 'GAN':
        final_dir = os.path.join(wkdir,'final')
    elif assign_method == 'Matching':
        final_dir = os.path.join(wkdir,'match_final')
    os.makedirs(final_dir,exist_ok=True)
    final_sdf = os.path.join(final_dir,f'{subjob_rank}_top{n_poses}_poses.sdf')
    # final_pdbqt = os.path.join(final_dir,f'{subjob_rank}_top{n_poses}_poses.pdbqt')
    if len(top_mols) > 0:
        sdf_writer = Chem.SDWriter(final_sdf)

        for mol in top_mols:
            sdf_writer.write(mol)
        sdf_writer.close()
    # os.system(f'obabel -isdf {final_sdf} -opdbqt -O {final_pdbqt} -p 7.4')

    t2 = time()
    print(f'assign time: {t1-t0}')
    print(f'scoring time: {t2-t1}')

if __name__ == "__main__":
    main()

