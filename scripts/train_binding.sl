#!/bin/sh

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

python /scratch/wdjo224/deep_protein_binding/src/train.py --exp_name=binding_debug_$(date +%T) --readout_dim=128 --T=3 --loss=bce --output_type=class --n_workers=1 --n_train_process=6 --output_dim=2 --target=label --lr=1e-3 --batch_size=256 --n_epochs=10 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --train_idxs=/scratch/wdjo224/deep_protein_binding/src/train.npy --val_idxs=/scratch/wdjo224/deep_protein_binding/src/val.npy
