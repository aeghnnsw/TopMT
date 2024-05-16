#!/usr/bin/env python3
import argparse
import os
import pickle
from glob import glob

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wkdir', type=str, \
                        help='Working directory')
    parser.add_argument('--top_n', type=int, \
                        help='Number of top molecules to select')
    parser.add_argument('--score_dir', type=str, default=None,\
                        help='Directory containing score files')
    parser.add_argument('--save_path', type=str, default=None,\
                        help='Path to save selected molecules')
    parser.add_argument('--match', action='store_true', default=False,\
                        help='whether the match mode')
    parser.add_argument('--n_jobs', type=int, default=None,\
                        help='Number of subjobs')
    args = parser.parse_args()
    if args.score_dir is None:
        args.score_dir = os.path.join(args.wkdir, 'scores')
    assert os.path.exists(args.score_dir), \
        f'Score directory does not exist: {args.score_dir}'
    score_files = glob(os.path.join(args.score_dir, '*.pkl'))
    score_files = sorted(score_files)
    if args.n_jobs is None:
        n_subjobs = len(score_files)
    else:
        n_subjobs = args.n_jobs
    scores = []
    for score_file in score_files:
        with open(score_file, 'rb') as f:
            score = pickle.load(f)
        scores.extend(score)
    scores_sorted = sorted(scores, key=lambda x: x[3])
    mol_list = []
    prefix = 'mols_match' if args.match else 'mols_assign'
    for i in tqdm(range(args.top_n)):
        subjob_rank = scores_sorted[i][0]
        batch_id = scores_sorted[i][1]
        mol_id = scores_sorted[i][2]
        score = scores_sorted[i][3]
        mol_name = f'mol_{subjob_rank}_{batch_id:03d}_{mol_id:04d}'
        mol_file = f'{prefix}_{subjob_rank+1}_of_{n_subjobs}/mol_batch_{batch_id:03d}.sdf'
        mol_file = os.path.join(args.wkdir,mol_file)
        sdf_suppl = Chem.SDMolSupplier(mol_file)
        mol_temp = sdf_suppl[int(mol_id)]
        assert mol_name == mol_temp.GetProp('_Name'), f'{mol_name} != {mol_temp.GetProp("_Name")}'
        mol_temp.SetProp('vina_score',str(score))
        mol_list.append(mol_temp)
    
    w = Chem.SDWriter(args.save_path)
    for mol in mol_list:
        w.write(mol)
    w.close()

if __name__ == '__main__':
    main()
