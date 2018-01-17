'''
    This script takes the mol2 files and converts to a smiles representation.

    by Derek Jones (01/16/17)

    # rdkit does not support reading mol2 files with more than one molecule, using this code as a reference on how
    to handle this: https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg01510.html
'''

import os
import argparse
from rdkit import Chem
import sys
import rdkit.Chem.Descriptors as descr

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str, help="path to directory containing mol2 files")
parser.add_argument('-o', type=str, help="path to directory to output smiles files")

args = parser.parse_args()



def RetrieveMol2Block(fileLikeObject, delimiter="@<TRIPOS>MOLECULE"):
    """generator which retrieves one mol2 block at a time
    """
    mol2 = []
    for line in fileLikeObject:
        if line.startswith(delimiter) and mol2:
            yield "".join(mol2)
            mol2 = []
        mol2.append(line)
    if mol2:
        yield "".join(mol2)


if __name__ == "__main__":
    with open(sys.argv[1]) as fi:
        for mol2 in RetrieveMol2Block(fi):
            rdkMolecule = Chem.MolFromMol2Block(mol2)
            print(descr.MolWt(rdkMolecule))
    for mol_file in os.listdir(args.i):
        smiles = Chem.MolToSmiles(mol_file)
        print(smiles)


