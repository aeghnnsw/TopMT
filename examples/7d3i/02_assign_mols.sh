#!/bin/bash
../../scripts/assign_from_tops.py \
        --ori_lig_file ./structures/7d3i_l.mol2 \
        --pocket_pdbqt ./structures/7d3i_p.pdbqt \
        --wkdir ./sample_pd \
        --assign_method GAN \
        --mol_assign_ckpt_path ../../saved_models/type_assign/checkpoints \
        --n_assigns 128 \
        --n_subjobs 10 \
        --subjob_rank 0 \
        --vina_threshold -8 \
        --save_batch_size 10000 \
        --n_process 4
