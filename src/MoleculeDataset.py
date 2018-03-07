import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer


class MoleculeDatasetCSV(Dataset):

    def __init__(self, csv_file, target, corrupt_path, cuda=False, scaling=None):
        super(MoleculeDatasetCSV, self).__init__()
        self.csv_file = csv_file
        self._cuda = cuda
        self.scaling = scaling
        self.target = target
        cols = ["receptor", "drugID", "smiles", "label"] + [target]

        self.data = pd.read_csv(csv_file, usecols=cols)
        self.corrupt_compound_df = pd.read_csv(corrupt_path)
        self.data = self.data[~self.data.drugID.isin(self.corrupt_compound_df.drugID)]
        if self.scaling == "std":
            print("standardizing data...")
            self.data[self.target] = StandardScaler().fit_transform(self.data[self.target])
        elif self.scaling == "norm":
            print("normalizing data...")
            self.data[self.target] = Normalizer().fit_transform(self.data[self.target])

        self.activities = self.data["label"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        # TODO: should an ordered dict be used here if number of targets becomes large?
        compound = self.data.iloc[item]
        item_dict = {}
        item_dict["smiles"] = compound["smiles"]
        item_dict[self.target] = torch.from_numpy(np.asarray([compound[self.target]], dtype=float)).float()


        return item_dict


class MoleculeDatasetH5(Dataset):
    def __init__(self, data_dir, list_dir, corrupt_path, target, num_workers, cuda=False):
        super(MoleculeDatasetH5, self).__init__()
        self.num_workers = num_workers
#         TODO: make sure targets is iterable
        self.targets = target

        self._cuda= cuda
        self.fo_dict = {}
        self.compound_df = pd.DataFrame()
        self.corrupt_compound_df = pd.read_csv(corrupt_path)
        self.fo = h5py.File("/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets/kinase_no_duplicates_with_smiles_fgfr1_1516755130.5441012.h5", "r", swmr=True)
        for file in os.listdir(list_dir):
            self.compound_df = pd.concat([self.compound_df, pd.read_csv(list_dir+"/"+file)])
        self.compound_df = pd.read_csv(list_dir+"/fgfr1.csv")

        #remove the precomputed corrupted inputs
        self.compound_df = self.compound_df[~self.compound_df["drugID"].isin(self.corrupt_compound_df.drugID)]        # shuffle the entries of the dataframe so compounds with common target are not grouped together sequentially
        self.compound_df = shuffle(self.compound_df)

        self.activities = self.compound_df["label"]

    def __len__(self):
        return self.compound_df.shape[0]

    def __getitem__(self, item):

        # get entry from compound df
        compound_row = self.compound_df.iloc[item]
        target_list = []
        receptor = compound_row["receptor"]
        drugID = compound_row["drugID"]

        # build up the target vector
        for target in self.targets:
            target_list.append(self.fo[receptor][drugID][target][0])

        # then get the smiles string, process it and then return its feature vectors

        data = self.construct_multigraph(self.fo[receptor][drugID]["smiles"][()][0])

        return {"g": data[0], "h": data[1],
                 "target": torch.from_numpy(np.asarray(target_list).astype('float')).float()}


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp = mp.get_context("forkserver")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nworkers", type=int, help="number of workers to use in data loader", default=1)
    parser.add_argument("--batch_size", type=int, help="batch size to use in data loader", default=1)
    args = parser.parse_args()

    print("{:=^100}".format(' Testing Dataloader '))
    data = MoleculeDatasetH5(data_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets", list_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/dataset_compounds",
                             corrupt_path="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv",targets=["label"],num_workers=1)
    print("size of dataset: {}".format(len(data)))
    from torch.utils.data import DataLoader

    def collate_fn(batch):

        return batch

    batch_size = args.batch_size
    num_workers = args.nworkers
    num_iters = int(np.ceil(len(data)/batch_size))
    mydata = DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    print("batch size: {} \t num_iterations: {} \t num_workers: {}".format(batch_size, num_iters, num_workers))
    for idx, batch in tqdm(enumerate(mydata), total=num_iters):
        pass

