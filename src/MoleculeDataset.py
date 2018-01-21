'''
    Code adapted from the Kesier Lab implementation of "Convolutional Networks on Graphs for Learning Molecular Fingerprints"

    url: https://github.com/keiserlab/keras-neural-graph-fingerprint
'''
import torch
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from preprocessing import tensorize_smiles, clean_data
from multiprocessing import cpu_count

# TODO: create a dataset class that loads the h5 file instead to get something more memory efficient
# TODO: generalize the label loading to allow for arbitrary number of labels
# TODO: write a method that returns tuple of dimensionalities

class MoleculeDataset(Dataset):
    def __init__(self, csv_file, labels, nrows=None, num_workers=1):
        self.csv_file = csv_file
        self.nrows = nrows
        self.num_workers = num_workers
        self.csv_data = None
        self.data = None
        self.labels = labels

    def read_csv_data(self):
        self.csv_data = pd.read_csv(self.csv_file, nrows=self.nrows)

    def clean_csv_data(self):
        self.csv_data = clean_data(self.csv_data, workers=self.num_workers)

    def tensorize_smiles(self):
        self.data = tensorize_smiles(self.csv_data["smiles"].as_matrix())
        self.labels = self.csv_data[self.labels].as_matrix()

    def __len__(self):
        assert self.data is not None and self.labels is not None
        return self.data[0].shape[0]

    def __getitem__(self, item):
        assert self.data is not None and self.labels is not None
        return {"atom": self.data[0][item].astype('float'), "bond": self.data[1][item].astype('float'),
                "edge": self.data[2][item].astype('float'), "target": self.labels[item].astype('float')}




class MoleculeDatasetH5(Dataset):
    def __init__(self, file, features, split, labels, num_workers):
        super(MoleculeDatasetH5, self).__init__()
        self.fo = h5py.File(file,"r")
        self.num_workers = num_workers
        '''
            iterate over each of the keys in the split of the dataset, using a dictionary store each of these lengths 
            separately, add all of these up to get the length of the full dataset. Use a helper function to do this?
        '''
        self.len_dict = {}
        self.len = None
        self.features = features
        self.labels = labels
        self.split = split

    def get_kinase_from_index(self, item):
        # iterate over the sizes associated with each kinase, if item > kinase size, then value must be in next kinase,
        # if item < kinase_size and > cumsum(kinase_sizes), then subtract item from cumsum(kinase_sizes) and return the
        # item at that result
        return item

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        '''
            upon reciept of the item index, use helper function to find the kinase dataset that the index belongs to
        :param item:
        :return:
        '''

        data = np.ndarray()
        for key in self.features:
            np.concatenate((data,h5py[]))

if __name__ == "__main__":

    print("{:=^100}".format(' Data preprocessing '))
    molecules = MoleculeDataset("/u/vul-d1/scratch/wdjo224/deep_protein_binding/test.csv")
    molecules.read_csv_data()
    molecules.clean_csv_data()
    molecules.tensorize_smiles()
