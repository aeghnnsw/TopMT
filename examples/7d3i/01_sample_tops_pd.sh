#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
sample_tops.py \
        --mode pocket \
        --ori_lig_file ./structures/7d3i_l.mol2 \
        --pocket_file ./structures/pocket.pdb \
        --wkdir ./sample_pd \
        --DGT_ckpt_path ../../saved_models/top_sample/checkpoints \
        --grid_boundary 0.01 \
        --n_max_spheres 80 \
        --sample_repeats 10 \
        --sample_tol 10000 \
        --natoms_upper 3 \
        --natoms_lower 3 \
        --n_tops 100 \
        --top_freq_lib "../../tops_freq_drug_bank.pkl" \
        --batch_size 64 \
        --threshold 0.3 \
        --search_steps 150 \
        --search_repeats 10
