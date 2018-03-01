#!/usr/bin/env bash

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

python src/get_hidden_features.py --loss=bce --output_type=class --output_dim=2 --model_path=/scratch/wdjo224/deep_protein_binding/experiments/binding/binding_1519359692.705323/checkpoints/binding_1519359692.705323_epoch4_step_4268 --target_list=label --exp_name=binding_test --batch_size=300 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --test_idxs=/scratch/wdjo224/deep_protein_binding/src/test.npy