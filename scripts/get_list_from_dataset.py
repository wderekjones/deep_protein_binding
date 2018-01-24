'''
    The purpose of this script is to read the h5 datasets and to generate a list of the compounds,
    along with their binding activity
    by: Derek Jones 01/24/2018

'''
import os
import h5py
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="input path to dataset directory")
parser.add_argument("--list_dir", type=str, help="input path to dataset list directory")

args = parser.parse_args()

# TODO: get a column of the target names, should just copy the target as many times as there are compounds
# TODO: get a column of the compound (target-drug molecule) activity, this requires looping over all drugs and getting the labels

if __name__ == "__main__":

    for dataset in tqdm(os.listdir(args.data_dir)):
        drug_dict = {}
        fo = h5py.File(args.data_dir+"/"+dataset, "r")
        root_key = list(fo)[0]
        drug_list = list(fo[root_key].keys())

        drug_dict[root_key] = drug_list
        output_df = pd.DataFrame(drug_dict)
        output_df.to_csv(args.list_dir+str(root_key)+".csv", index=False, header=None)
