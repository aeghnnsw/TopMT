#!/bin/bash
source /data/shen//PBDD/config
assign_from_tops.py \
        --ori_lig_file ./structures/7d3i_l.mol2 \
        --pocket_pdbqt ./structures/7d3i_p.pdbqt \
        --wkdir ./sample_pd \
        --assign_method GAN \
        --mol_assign_ckpt_path ../../saved_models/type_assign/checkpoints \
        --n_assigns 128 \
        --frag_lib_path ../frag_lib_259k.pkl \
        --n_subjobs $1 --subjob_rank $2 \
        --vina_threshold -8 \
        --save_batch_size 10000 \
