#!/usr/bin/env python3
import argparse
import os
import sys
from rdkit import Chem
from tqdm import tqdm
from glob import glob
import multiprocessing

def copy_pos_file(mol_name,args:argparse.Namespace):
    identifiers = mol_name.split('_')
    job_id = int(identifiers[1])
    batch_id = int(identifiers[2])
    mol_id = int(identifiers[3])
    pos_dir = os.path.join(args.wkdir, f'pos_{job_id}')
    pos_files = glob(os.path.join(pos_dir, '*.pdbqt'))
    mol_file_start = f'mol_{batch_id:03d}_{mol_id:04d}'
    find = False
    for pos_file_temp in pos_files:
        if mol_file_start in pos_file_temp:
            pos_file = pos_file_temp
            find = True
            break
    pos_file_name = os.path.basename(pos_file)
    identifier = pos_file_name.split('_')
    identifier.insert(1,str(job_id))
    new_name = '_'.join(identifier)
    new_pos_file = os.path.join(args.new_pos_dir, new_name)
    # copy pos file to new directory
    os.system(f'cp {pos_file} {new_pos_file}')
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wkdir', type=str, \
                        help='Working directory')
    parser.add_argument('--keep_mols_file', type=str, required=True,\
                        help='File containing molecules to keep')
    parser.add_argument('--new_pos_dir', type=str, required=True,\
                        help='Directory to store new pos files')
    args = parser.parse_args()
    
    os.makedirs(args.new_pos_dir, exist_ok=True)
    # read mols to keep
    assert os.path.exists(args.keep_mols_file), \
        'File containing molecules to keep does not exist'
    sdf_suppl = Chem.SDMolSupplier(args.keep_mols_file)
    task_list = []
    for mol_temp in sdf_suppl:
        mol_name = mol_temp.GetProp('_Name')
        task_list.append([mol_name,args])
    print('number of molecules to keep: ', len(task_list))
    # run in parallel
    n_cpus = multiprocessing.cpu_count()
    with multiprocessing.get_context("fork").Pool(n_cpus) as pool:
        pool.starmap(copy_pos_file, task_list)

if __name__ == '__main__':
    main()

