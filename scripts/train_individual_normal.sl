#!/bin/sh

#SBATCH --array=0-12

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

A=('Uc' 'Ui' 'Hy' 'TPSA(NO)' 'MLOGP' 'MLOGP2' 'SAtot' 'SAacc' 'SAdon' 'Vx' 'VvdwMG' 'VvdwZAZ' 'PDI')

python /scratch/wdjo224/deep_protein_binding/src/train.py --loss=normal --target_list=${A[$SLURM_ARRAY_TASK_ID]} --exp_name=${A[$SLURM_ARRAY_TASK_ID]}_normal --lr=1e-3 --batch_size=300 --n_epochs=2 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --train_idxs=/scratch/wdjo224/deep_protein_binding/src/train.npy --val_idxs=/scratch/wdjo224/deep_protein_binding/src/val.npy