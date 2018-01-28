#!/bin/sh

source activate deep_protein_binding

python /scratch/wdjo224/deep_protein_binding/src/preprocessing.py -i /scratch/wdjo224/data/data/kinase/with_pocket/kinase_no_duplicates_with_smiles.csv -o /scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv
