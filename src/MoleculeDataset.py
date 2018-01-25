'''
    Code adapted from the Kesier Lab implementation of "Convolutional Networks on Graphs for Learning Molecular Fingerprints"

    url: https://github.com/keiserlab/keras-neural-graph-fingerprint
'''
import torch
import os
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from preprocessing import tensorize_smiles, tensorize_smiles_job, clean_data

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
    def __init__(self, data_dir, list_dir, targets, num_workers):
        super(MoleculeDatasetH5, self).__init__()
        self.num_workers = num_workers
        # TODO: make sure targets is iterable
        self.targets = targets
        self.fo_dict = {}
        self.compound_df = pd.DataFrame()
        # from the input file list, open each of the binary files (read mode) and store in a dictionary
        for file in os.listdir(data_dir):
            fo = h5py.File(data_dir+"/"+file,"r")
            key = list(fo.keys())[0]
            self.fo_dict[key] = fo
        for file in os.listdir(list_dir):
            self.compound_df = pd.concat([self.compound_df, pd.read_csv(list_dir+"/"+file)])

        # shuffle the entries of the dataframe so compounds with common target are not grouped together sequentially
        self.compound_df = shuffle(self.compound_df)

    def __len__(self):
        return self.compound_df.shape[0]

    def __getitem__(self, item):

        # get entry from compound df
        compound_row = self.compound_df.loc[item]
        target_list = []
        receptor = compound_row["receptor"]
        drugID = compound_row["drugID"]
        # build up the target vector
        for target in self.targets:
            target_list.append(self.fo_dict[receptor][receptor][drugID][target][0])

        # then get the smiles string, process it and then return its feature vectors
        data = tensorize_smiles_job(self.fo_dict[receptor][receptor][drugID]["smiles"][()])
        assert data is not None and target_list is not None
        return {"atom": data[0].astype('float'), "bond": data[1].astype('float'),
                "edge": data[2].astype('float'), "target": np.asarray(target_list).astype('float')}

if __name__ == "__main__":

    print("{:=^100}".format(' Data preprocessing '))
    data = MoleculeDatasetH5("/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets", "/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/dataset_compounds", ["label"],1)
    print("size of dataset: {}".format(len(data)))
    from torch.utils.data import DataLoader


    def collate_fn(batch):
        # Note that batch is a list
        # batch = list(map(list, zip(*batch)))  # transpose list of list
        # out = None
        # You should know that batch[0] is a fixed-size tensor since you're using your customized Dataset
        # reshape batch[0] as (N, H, W)
        # batch[1] contains tensors of different sizes; just let it be a list.
        # If your num_workers in DataLoader is bigger than 0
        #     numel = sum([x.numel() for x in batch[0]])
        #     storage = batch[0][0].storage()._new_shared(numel)
        #     out = batch[0][0].new(storage)
        # batch[0] = torch.stack(batch[0], 0, out=out)
        return batch

    mydata = DataLoader(data, batch_size=10, num_workers=1, collate_fn=collate_fn)

    for idx, batch in enumerate(mydata):
        print("now loading batch {}".format(idx))
