#!/usr/bin/env bash

#SBATCH --array=0-35

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

T=(3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1)
R=(10 20 30 40 50 60 70 80 90 100 110 120 10 20 30 40 50 60 70 80 90 100 110 120 10 20 30 40 50 60 70 80 90 100 110 120)
python /scratch/wdjo224/deep_protein_binding/src/train.py --exp_name=${R[$SLURM_ARRAY_TASK_ID]}_${T[$SLURM_ARRAY_TASK_ID]}_binding --readout_dim=${R[$SLURM_ARRAY_TASK_ID]} --T=${T[$SLURM_ARRAY_TASK_ID]} --loss=bce --output_type=class --n_workers=1 --n_train_process=8 --output_dim=2 --target=label --lr=1e-3 --batch_size=256 --n_epochs=10 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --train_idxs=/scratch/wdjo224/deep_protein_binding/src/train.npy --val_idxs=/scratch/wdjo224/deep_protein_binding/src/val.npy


