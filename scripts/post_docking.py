#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import pickle
from glob import glob
from time import time

from dimorphite_dl import DimorphiteDL
import numpy as np
from pbdd.post_processing.scoring import vina_docking
from pbdd.post_processing.utils import convert_rdkit_pdbqt_str, protonate_mol
from rdkit import Chem
from tqdm import tqdm


def main():
    t0 = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_lig_file', type=str,\
                        help='path to the original ligand file')
    parser.add_argument('--pocket_pdbqt',type=str,\
                        help='path to the pocket pdbqt file')
    parser.add_argument('--wkdir',type=str,\
                        help='path to the working directory')

    parser.add_argument('--mols_to_dock',type=str,\
                        help='file or directory of the molecules to dock')
    parser.add_argument('--n_subjobs',type=int,default=1,\
                        help='number of subjobs')
    parser.add_argument('--subjob_rank',type=int,default=0,\
                        help='rank of the subjob')
    parser.add_argument('--n_processes',type=int,default=0,\
                        help='number of processes for docking')
    parser.add_argument('--vina_threshold',type=float,default=-8,\
                        help='vina score threshold for saving the pose')
    parser.add_argument('--box_length',type=float,default=30.0,\
                        help='box length for docking')
    parser.add_argument('--write_pose',action='store_true',\
                        help='whether to write the pose')
    parser.add_argument('--redock',action='store_true',\
                        help='whether to redock the molecules')
    parser.add_argument('--exhaustiveness',type=int,default=16,\
                        help='exhaustiveness for docking')
    parser.add_argument('--rename',action='store_true',\
                        help='whether to rename the ligand')
    parser.add_argument('--show_prog',action='store_true',default=False,\
                        help='whether to show the progress bar')
    args = parser.parse_args()

    # read pocket pdbqt and original ligand
    assert os.path.exists(args.ori_lig_file),f'{args.ori_lig_file} does not exist'
    assert os.path.exists(args.pocket_pdbqt),f'{args.pocket_pdbqt} does not exist'
    lig_file_type = args.ori_lig_file.split('.')[-1]
    if lig_file_type=='pdb':
        ori_mol = Chem.MolFromPDBFile(args.ori_lig_file, removeHs=False)
        ori_pdbqt = args.ori_lig_file.replace('.pdb','.pdbqt')
    elif lig_file_type=='mol2':
        ori_mol = Chem.MolFromMol2File(args.ori_lig_file, removeHs=False)
        ori_pdbqt = args.ori_lig_file.replace('.mol2','.pdbqt')
    else:
        raise ValueError(f'unsupported ligand file type: {lig_file_type}')
    pos = ori_mol.GetConformer().GetPositions()
    pos_min = np.min(pos,axis=0)
    pos_max = np.max(pos,axis=0)
    lig_pos_center = (pos_min+pos_max)/2

    # Redock options
    if args.redock:
        if not os.path.exists(ori_pdbqt):
            os.system(f'prepare_ligand -l {args.ori_lig_file} -A hydrogens')
        redock_name = ori_pdbqt[:-6]
        print(redock_name)
        vina_docking(ori_pdbqt,args.pocket_pdbqt,lig_pos_center,save_name=redock_name,\
                    save_threshold=0,box_length=args.box_length,write_pose=True,\
                    exhaustiveness=args.exhaustiveness)
        return None

    # create wkdir if not exists
    os.makedirs(args.wkdir,exist_ok=True)

    # read sdf 
    if os.path.isdir(args.mols_to_dock):
        sdf_files = glob(os.path.join(args.mols_to_dock,'*.sdf'))
        # sort the pdbqt files by name
        sdf_files = sorted(sdf_files)
    else:
        sdf_files = [args.mols_to_dock]
    
    # read all sdf files
    rdkit_mols = []
    for sdf_file in sdf_files:
        rdkit_mol = Chem.SDMolSupplier(sdf_file)
        rdkit_mols.extend(rdkit_mol)
    n_mols = len(rdkit_mols)
    
    # split rdkit mols into subjobs
    batch_size = n_mols//args.n_subjobs + 1 
    start = args.subjob_rank*batch_size
    end = min((args.subjob_rank+1)*batch_size,n_mols)
    subjob_mol_list = rdkit_mols[start:end]
    print(f'all molecules to dock: {n_mols}')
    print(f'subjob {args.subjob_rank} molecules to dock: {len(subjob_mol_list)}')

    if len(subjob_mol_list)==0:
        return None
    # assign correct protonated states of the ligands
    print('protonating molecules')
    protonated_mols = []
    # if args.show_prog:
        # subjob_mol_list = tqdm(subjob_mol_list)
    for mol in tqdm(subjob_mol_list):
        protonated_mols.extend(protonate_mol(mol))
    # with multiprocessing.get_context('fork').Pool(8) as pool:
    #     protonated_mol_list = pool.map(protonate_mol,subjob_mol_list)
    # for protonated_temps in protonated_mol_list:
    #     if protonated_temps is not None:
    #         protonated_mols.extend(protonated_temps)
    # convert rdkit mols to pdbqt 
    
    # save protonated mols for visualization
    protonated_mol_sdf = os.path.join(args.wkdir,f'protonated_mols_{args.subjob_rank:04d}.sdf')
    sd_writer = Chem.SDWriter(protonated_mol_sdf)
    for mol in protonated_mols:
        sd_writer.write(mol)
    sd_writer.close()

    subjob_pdbqt_list = []
    lig_idx = 0
    if args.show_prog:
        protonated_mols = tqdm(protonated_mols)
    for mol in protonated_mols:
        if mol is None:
            continue
        pdbqt_str = convert_rdkit_pdbqt_str(mol,add_h=True)
        if args.rename:
            mol_name = f'{args.subjob_rank}_{lig_idx:05d}'
            lig_idx += 1
        else:
            mol_name = mol.GetProp('_Name')
        subjob_pdbqt_list.append([pdbqt_str,mol_name])

    print(f'mols to dock after protonation: {len(subjob_pdbqt_list)}')
    # set up docking tasks
    task_list = []
    mol_ids = []
    dock_pos_dir = os.path.join(args.wkdir,'docking_poses')
    os.makedirs(dock_pos_dir,exist_ok=True)

    for pdbqt_str,mol_name in subjob_pdbqt_list:
        save_path = os.path.join(dock_pos_dir,mol_name)
        task_list.append([pdbqt_str,args.pocket_pdbqt,lig_pos_center,save_path,\
                          args.vina_threshold,args.box_length,args.write_pose,\
                          args.exhaustiveness])
        mol_ids.append(mol_name)

    n_cpus = os.cpu_count()
    # make sure each process at least has 12 cpus
    if args.n_processes>0:
        n_process = min(n_cpus//8,args.n_processes)
    else:
        n_process = n_cpus//8
    with multiprocessing.get_context('fork').Pool(n_process) as pool:
        scores = pool.starmap(vina_docking,task_list)

    # save results to a file
    assert len(scores)==len(mol_ids)
    # some mol_ids have multiple protonated states, keep the one with the lowest score

    mol_score_dict = {}
    for mol_id,score in zip(mol_ids,scores):
        if mol_id not in mol_score_dict:
            mol_score_dict[mol_id] = score
        else:
            if score<mol_score_dict[mol_id]:
                mol_score_dict[mol_id] = score

    mol_ids = list(mol_score_dict.keys())
    scores = [mol_score_dict[mol_id] for mol_id in mol_ids]
    
    docking_score_dir = os.path.join(args.wkdir,'docking_scores')
    os.makedirs(docking_score_dir,exist_ok=True)
    docking_score_file = os.path.join(docking_score_dir,f'docking_scores_{args.subjob_rank}.pkl')
    with open(docking_score_file,'wb') as f:
        pickle.dump([mol_ids,scores],f)
    t2 = time()
    print(f'subjob {args.subjob_rank} docking time: {t2-t0:.2f} s')

if __name__ == "__main__":
    main()

