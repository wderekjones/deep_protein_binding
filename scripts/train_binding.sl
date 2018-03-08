#!/bin/sh

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

python /scratch/wdjo224/deep_protein_binding/src/train.py --loss=bce --output_type=class --n_workers=0 --n_train_process=8 --output_dim=2 --exp_name=binding --target=label --lr=1e-3 --batch_size=64 --n_epochs=2 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --train_idxs=/scratch/wdjo224/deep_protein_binding/src/train.npy --val_idxs=/scratch/wdjo224/deep_protein_binding/src/val.npy