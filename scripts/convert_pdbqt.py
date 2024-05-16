#!/usr/bin/env python3
import argparse
import multiprocessing
import os
from glob import glob


def convert_process(sdf_file):
    pdbqt_file = sdf_file.replace('.sdf','.pdbqt')
    os.system(f'obabel -isdf {sdf_file} -opdbqt -O {pdbqt_file} -p 7.4')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_file', type=str,\
                        help='path to the sdf file or folder')
    parser.add_argument('--multiprocess', action='store_true',\
                        help='whether to use multiprocess')
    args = parser.parse_args()
    sdf_file = args.sdf_file
    if os.path.isdir(sdf_file):
        sdf_files = glob(os.path.join(sdf_file,'*.sdf'))
    else:
        sdf_files = [sdf_file]
       
    if args.multiprocess:       
        n_cpus = multiprocessing.cpu_count()
        with multiprocessing.get_context('spawn').Pool(n_cpus) as pool:
            pool.map(convert_process,sdf_files)
    else:
        for sdf_file in sdf_files:
            pdbqt_file = sdf_file.replace('.sdf','.pdbqt')
            os.system(f'obabel -isdf {sdf_file} -opdbqt -O {pdbqt_file} -p 7.4')

if __name__ == "__main__":
    main()