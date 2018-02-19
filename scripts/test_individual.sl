#!/bin/sh

#SBATCH --array=0-12

source activate deep_protein_binding
export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24

A=('Uc' 'Ui' 'Hy' 'TPSA(NO)' 'MLOGP' 'MLOGP2' 'SAtot' 'SAacc' 'SAdon' 'Vx' 'VvdwMG' 'VvdwZAZ' 'PDI')
B=('/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570837.635661/checkpoints/debug_1518570837.635661_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4549797/checkpoints/debug_1518570839.4549797_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4412563/checkpoints/debug_1518570839.4412563_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.453774/checkpoints/debug_1518570839.453774_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4417353/checkpoints/debug_1518570839.4417353_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4413881/checkpoints/debug_1518570839.4413881_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4425485/checkpoints/debug_1518570839.4425485_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4508493/checkpoints/debug_1518570839.4508493_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.444143/checkpoints/debug_1518570839.444143_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.446972/checkpoints/debug_1518570839.446972_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4452367/checkpoints/debug_1518570839.4452367_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570839.4419758/checkpoints/debug_1518570839.4419758_epoch1_step_1667' \
'/scratch/wdjo224/deep_protein_binding/experiments/debug/debug_1518570843.2404766/checkpoints/debug_1518570843.2404766_epoch1_step_1667')
python /scratch/wdjo224/deep_protein_binding/src/test.py --model_path=${B[$SLURM_ARRAY_TASK_ID]} --target_list=${A[$SLURM_ARRAY_TASK_ID]} --exp_name=${A[$SLURM_ARRAY_TASK_ID]} --batch_size=300 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --test_idxs=/scratch/wdjo224/deep_protein_binding/src/test.npy