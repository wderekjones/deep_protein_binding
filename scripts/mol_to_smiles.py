'''
    This script takes the mol2 files and converts to a smiles representation.

    by Derek Jones (01/16/17)

    # rdkit does not support reading mol2 files with more than one molecule, using this code as a reference on how
    to handle this: https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg01510.html
'''

import os
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import sys
import rdkit.Chem.Descriptors as descr

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str, help="path to directory containing mol2 files")
parser.add_argument('-o', type=str, help="path to directory to output smiles files")

args = parser.parse_args()


# TODO: need to extract the name of the molecule and then pair this with the smiles representation...then store in a dataframe...then output to .csv


def retrieve_mol2block(fileLikeObject, delimiter="@<TRIPOS>MOLECULE"):
    """generator which retrieves one mol2 block at a time
    """

    # directly after the line @<TRIPOS>MOLECULE contains the name of the molecule

    molname = None
    prevline = ""
    mol2 = []
    for line in fileLikeObject: # line will contain the molecule name followed by a newline character
        if line.startswith(delimiter) and mol2:
            yield (molname.strip("/n").replace('-','_'), "".join(mol2))
            molname = ""
            mol2 = []
        elif prevline.startswith(delimiter):
            molname = line
        mol2.append(line)
        prevline = line
    if mol2:
        yield (molname, "".join(mol2))
        molname = ""


if __name__ == "__main__":
    output_df = pd.DataFrame()
    smiles_list = []
    molname_list = []
    for mol_file in tqdm(os.listdir(args.i)): # could implement multiprocessing here to do this in parallel to build the lists

        with open(args.i+"/"+mol_file) as fi:
            for molname, mol2 in retrieve_mol2block(fi):
                # need to better understand the consequences of using the sanitize=False setting, https://www.wildcardconsulting.dk/useful-information/the-good-the-bad-and-the-ugly-rdkit-molecules/
                rdkMolecule = Chem.MolFromMol2Block(mol2, sanitize=False)
                smiles = Chem.MolToSmiles(rdkMolecule)
                molname_list.append(molname)
                smiles_list.append(smiles)

    output_df["molname"] = molname_list
    output_df["smiles"] = smiles_list
    output_df.to_csv(args.o, index=False)
