import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
import dc_features as dc
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer
from rdkit import Chem


class MoleculeDataset(Dataset):
    def __init__(self,cuda=False):
        super(MoleculeDataset,self)
        self._cuda = cuda

    def construct_multigraph(self, smile):
        g = {}
        h = {}
        molecule = Chem.MolFromSmiles(smile)
        for i in range(0, molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i)
            if self._cuda:
                h[i] = Variable(
                    torch.from_numpy(dc.atom_features(atom_i)).view(1, 75)).float().cuda()
            else:
                h[i] = Variable(
                    torch.from_numpy(dc.atom_features(atom_i)).view(1, 75)).float()

            for j in range(0, molecule.GetNumAtoms()):
                e_ij = molecule.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    # e_ij = map(lambda x: 1 if x == True else 0,
                    #             dc.feat.graph_features.bond_features(e_ij))  # ADDED edge feat
                    e_ij = map(lambda x: 1 if x == True else 0,
                        dc.bond_features(e_ij))
                    if self._cuda:
                        e_ij = Variable(torch.from_numpy(np.fromiter(e_ij, dtype=float))).view(1, 6).float().cuda()
                    else:
                        e_ij = Variable(torch.from_numpy(np.fromiter(e_ij, dtype=float))).view(1, 6).float()
                    atom_j = molecule.GetAtomWithIdx(j)
                    if i not in g:
                        g[i] = []
                        g[i].append((e_ij, j))

        return g, h

    def __len__(self):
        raise Exception("Not implemented for abstract class")

    def __getitem__(self, item):
        raise Exception("Not implemented for abstract class")


class MoleculeDatasetCSV(MoleculeDataset):

    def __init__(self, csv_file, targets, corrupt_path, cuda=False, scaling=None):
        super(MoleculeDatasetCSV, self).__init__()
        self.csv_file = csv_file
        self._cuda = cuda
        self.scaling = scaling
        self.targets = targets
        cols = ["receptor", "drugID", "smiles", "label"] + targets
        self.data = pd.read_csv(csv_file, usecols=cols)
        self.corrupt_compound_df = pd.read_csv(corrupt_path)
        self.data = self.data[~self.data.drugID.isin(self.corrupt_compound_df.drugID)]
        if self.scaling == "std":
            print("standardizing data...")
            self.data[self.targets] = StandardScaler().fit_transform(self.data[self.targets])
        elif self.scaling == "norm":
            print("normalizing data...")
            self.data[self.targets] = Normalizer().fit_transform(self.data[self.targets])

        self.data = shuffle(self.data)
        self.activities = self.data["label"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        # TODO: should an ordered dict be used here if number of targets becomes large?
        compound = self.data.iloc[item]
        smiles = compound["smiles"]
        data = self.construct_multigraph(smiles)
        item_dict = {"g": data[0], "h": data[1]}
        for target in self.targets:
            item_dict[target] = np.asarray([compound[target]],dtype=float)

        return item_dict


class MoleculeDatasetH5(MoleculeDataset):
    def __init__(self, data_dir, list_dir, corrupt_path, targets, num_workers, cuda=False):
        super(MoleculeDatasetH5, self).__init__()
        self.num_workers = num_workers
#         TODO: make sure targets is iterable
        self.targets = targets
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
                 "target": np.asarray(target_list).astype('float')}


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
        # just here to take up space
        x = batch
