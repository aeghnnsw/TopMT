#!/usr/bin/env python3

import argparse
import os

from vina import Vina


def frag_dock(pocket_pdbqt,frag_pdbqt,pos_center,save_path,\
              exhaustiveness:int=32,\
              n_poses:int=20,box_length:float=25):
    try:
        v = Vina(verbosity=0)
        v.set_receptor(pocket_pdbqt)
        if os.path.exists(frag_pdbqt):
            v.set_ligand(frag_pdbqt)
        else:
            v.set_ligand_from_string(frag_pdbqt)
        v.compute_vina_maps(pos_center,[box_length,box_length,box_length])
        v.dock(exhaustiveness=exhaustiveness,n_poses=n_poses)
        v.write_poses(save_path,n_poses=n_poses,energy_range=5)
        return True
    except Exception as e:
        print(e)
        return False
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pocket_pdbqt',type=str,\
                        help='path to the pocket pdbqt file')
    parser.add_argument('--fraglib_pdbqt',type=str,\
                        help='path to the fragment library pdbqt file')
    parser.add_argument('--pos_center', nargs=3, type=float,\
                        help='position center of the pocket, provided as x y z')
    parser.add_argument('--save_dir',type=str,\
                        help='path to the save directory')
    parser.add_argument('--exhaustiveness',type=int,default=32,\
                        help='exhaustiveness for docking')
    parser.add_argument('--n_poses',type=int,default=20,\
                        help='number of poses to save')
    parser.add_argument('--box_length',type=float,default=25.0,\
                        help='box length for docking')
    args = parser.parse_args()
    
    assert os.path.exists(args.pocket_pdbqt),f'{args.pocket_pdbqt} does not exist'
    assert os.path.exists(args.fraglib_pdbqt),f'{args.fraglib_pdbqt} does not exist'
    os.makedirs(args.save_dir,exist_ok=True)
    frag_dock_dir = os.path.join(args.save_dir,'frag_dock')
    os.makedirs(frag_dock_dir,exist_ok=True)

    pos_center = args.pos_center
    frag_list = []
    with open(args.fraglib_pdbqt, 'r') as f:
        lines = f.readlines()
        mol_temp = ''
        for line in lines:
            if line.startswith('MODEL'):
                mol_temp = ''
                continue
            elif line.startswith('ENDMDL'):
                frag_list.append(mol_temp)
                continue
            mol_temp += line
    
    for i, frag_temp in enumerate(frag_list):
        save_pose_path = os.path.join(frag_dock_dir,f'frag_{i:03d}.pdbqt')
        frag_dock(args.pocket_pdbqt,frag_temp,\
                  pos_center,save_pose_path,\
                  exhaustiveness=args.exhaustiveness,\
                  n_poses=args.n_poses,\
                  box_length=args.box_length)
        


if __name__ == "__main__":
    main()