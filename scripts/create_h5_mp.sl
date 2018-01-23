#!/bin/bash -l

source activate deep_protein_binding

python create_h5_mp.py -i /scratch/wdjo224/data/data/kinase/with_pocket/kinase_no_duplicates_with_smiles.csv -o /scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles