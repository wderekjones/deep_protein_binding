'''
    Code adapted from the Kesier Lab implementation of "Convolutional Networks on Graphs for Learning Molecular Fingerprints"

    url: https://github.com/keiserlab/keras-neural-graph-fingerprint
'''
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from preprocessing import tensorize_smiles, clean_data
from multiprocessing import cpu_count

# TODO: create a dataset class that loads the h5 file instead to get something more memory efficient


class MoleculeDataset(Dataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.labels = None

    def read_csv_data(self, nrows=None):
        self.csv_data = pd.read_csv(self.csv_file, nrows=nrows)

    def clean_csv_data(self, workers=cpu_count()-1):
        self.csv_data = clean_data(self.csv_data, workers=workers)

    def tensorize_smiles(self):
        self.data = tensorize_smiles(self.csv_data["smiles"].as_matrix())
        self.labels = self.csv_data["active"].as_matrix()

    def __len__(self):
        assert self.data is not None and self.labels is not None
        return self.data[0].shape[0]

    def __getitem__(self, item):
        assert self.data is not None and self.labels is not None
        return {"atom": self.data[0][item].astype('float'), "bond": self.data[1][item].astype('float'),
                "edge": self.data[2][item].astype('float'), "target": self.labels[item].astype('float')}


if __name__ == "__main__":

    print("{:=^100}".format(' Data preprocessing '))
    molecules = MoleculeDataset("/u/vul-d1/scratch/wdjo224/deep_protein_binding/test.csv")
    molecules.read_csv_data()
    molecules.clean_csv_data(workers=cpu_count()-1)
    molecules.tensorize_smiles()
