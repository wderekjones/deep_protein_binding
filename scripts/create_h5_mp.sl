#!/bin/sh

#SBATCH -p GPU2
#SBATCH --array=0-25

A=(abl1 akt1 akt2 braf cdk2 csf1r egfr fak1 fgfr1 igf1r jak2 kit kpcb lck mapk2 met mk01 mk10 mk14 mp2k1 plk1 rock1 src tgfr1 vgfr2 wee1)

source activate deep_protein_binding

python create_h5_mp.py -i /scratch/wdjo224/data/data/kinase/with_pocket/kinase_no_duplicates_with_smiles.csv -o /scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles -k ${A[$SLURM_ARRAY_TASK_ID]}



























