#!/usr/bin/env python3
from rdkit import Chem, RDLogger
import argparse
import os
from glob import glob
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mols_to_sort', type=str, required=True,\
                        help='Path or directory to the file containing the molecules to sort')
    parser.add_argument('--output_dir', type=str, required=True,\
                        help='Path to the directory where the sorted molecules will be saved')
    parser.add_argument('--top_mols', type=int ,default=50000,\
                        help='Number of top molecules to save')
    parser.add_argument('--no_convert',action='store_true',\
                        help='Whether to convert the molecules to pdbqt format')
    args = parser.parse_args()

    assert os.path.exists(args.mols_to_sort),f'{args.mols_to_sort} does not exist'
    if os.path.isdir(args.mols_to_sort):
        mol_files = glob(os.path.join(args.mols_to_sort,'*.sdf'))
    elif os.path.isfile(args.mols_to_sort):
        mol_files = [args.mols_to_sort]
    mol_list = []
    for mol_file in mol_files:
        mols_temp = Chem.SDMolSupplier(mol_file)
        for mol_temp in tqdm(mols_temp):
            assert mol_temp.HasProp('_Name'),'mol does not have _Name'
            assert mol_temp.HasProp('vina_score'),f'{mol_temp.GetProp("_Name")} does not have vina_score'
            vina_score_temp = float(mol_temp.GetProp('vina_score'))
            mol_list.append([mol_temp,vina_score_temp])
    mol_list = sorted(mol_list,key=lambda x:x[1])
    if len(mol_list)>args.top_mols:
        top_list = mol_list[:args.top_mols]
    else:
        top_list = mol_list
    
    os.makedirs(args.output_dir,exist_ok=True)
    sort_sdf_file = os.path.join(args.output_dir,f'sorted_top{args.top_mols}.sdf')
    w = Chem.SDWriter(sort_sdf_file)
    for top_mol in top_list:
        w.write(top_mol[0])
    w.close()
    if not args.no_convert:
        pdbqt_file = sort_sdf_file.replace('.sdf','.pdbqt')
        os.system(f'obabel -isdf {sort_sdf_file} -opdbqt -O {pdbqt_file} -p 7.4')

if __name__ == "__main__":
    main()
