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


# these compounds don't have valid smiles representations, remove these after compound lists have been loaded
bad_mol_list=["fak1_decoy_450","kpcb_decoy_539", "tgfr1_decoy_3817", "igf1r_decoy_3764", "igf1r_decoy_3763"]


# class MoleculeDataset(Dataset):
#     def __init__(self, csv_file, labels, nrows=None, num_workers=1):
#         self.csv_file = csv_file
#         self.nrows = nrows
#         self.num_workers = num_workers
#         self.csv_data = None
#         self.data = None
#         self.labels = labels
#
#     def read_csv_data(self):
#         self.csv_data = pd.read_csv(self.csv_file, nrows=self.nrows)
#
#     def clean_csv_data(self):
#         self.csv_data = clean_data(self.csv_data, workers=self.num_workers)
#
#     def tensorize_smiles(self):
#         self.data = tensorize_smiles(self.csv_data["smiles"].as_matrix())
#         self.labels = self.csv_data[self.labels].as_matrix()
#
#     def __len__(self):
#         assert self.data is not None and self.labels is not None
#         return self.data[0].shape[0]
#
#     def __getitem__(self, item):
#         assert self.data is not None and self.labels is not None
#         return {"atom": self.data[0][item].astype('float'), "bond": self.data[1][item].astype('float'),
#                 "edge": self.data[2][item].astype('float'), "target": self.labels[item].astype('float')}


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
            fo_path = data_dir+"/"+file
            fo = h5py.File(fo_path,"r", libver="latest")
            key = list(fo.keys())[0]
            fo.close()
            self.fo_dict[key] = fo_path
        for file in os.listdir(list_dir):
            self.compound_df = pd.concat([self.compound_df, pd.read_csv(list_dir+"/"+file)])

        # remove the bad mol files
        for bad_mol in bad_mol_list:
            self.compound_df.drop(self.compound_df[self.compound_df["drugID"] == bad_mol].index, inplace=True)

        # shuffle the entries of the dataframe so compounds with common target are not grouped together sequentially
        self.compound_df = shuffle(self.compound_df)

    def __len__(self):
        return self.compound_df.shape[0]

    def __getitem__(self, item):

        # get entry from compound df
        compound_row = self.compound_df.iloc[item]
        target_list = []
        receptor = compound_row["receptor"]
        drugID = compound_row["drugID"]
        fo_path = self.fo_dict[receptor]
        fo = h5py.File(fo_path, "r", libver="latest")
        # build up the target vector
        for target in self.targets:
            target_list.append(fo[receptor][drugID][target][0])

        # then get the smiles string, process it and then return its feature vectors
        data = tensorize_smiles_job(fo[receptor][drugID]["smiles"][()])
        fo.close()
        assert data is not None and target_list is not None
        return {"atom": data[0].astype('float'), "bond": data[1].astype('float'),
                "edge": data[2].astype('float'), "target": np.asarray(target_list).astype('float')}

if __name__ == "__main__":

    print("{:=^100}".format(' Testing Dataloader '))
    data = MoleculeDatasetH5("/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets", "/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/dataset_compounds", ["label"],1)
    print("size of dataset: {}".format(len(data)))
    from torch.utils.data import DataLoader


    def collate_fn(batch):

        return batch

    mydata = DataLoader(data, batch_size=50, num_workers=6, collate_fn=collate_fn)

    for idx, batch in enumerate(mydata):
        print("now loading batch {}".format(idx))
