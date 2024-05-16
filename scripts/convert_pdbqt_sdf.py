#!/usr/bin/env python3

import argparse
import os
from glob import glob

from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbqt_dir', type=str, \
                        help='pdbqt file or Directory containing pdbqt files')
    args = parser.parse_args()
    assert os.path.exists(args.pdbqt_dir), \
        f'pdbqt file or Directory does not exist: {args.pdbqt_dir}'
    if os.path.isdir(args.pdbqt_dir):
        pdbqt_files = glob(os.path.join(args.pdbqt_dir, '*.pdbqt'))
    else:
        pdbqt_files = [args.pdbqt_dir]
    for pdbqt_file in pdbqt_files:
        pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file,skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        sdf_file = pdbqt_file.replace('.pdbqt', '.sdf')
        writer = Chem.SDWriter(sdf_file)
        for rdkit_mol in rdkit_mols:
            writer.write(rdkit_mol)
        writer.close()

if __name__ == '__main__':
    main()