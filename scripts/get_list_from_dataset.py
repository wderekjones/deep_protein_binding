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

if __name__ == "__main__":

    print("reading datasets..")
    for dataset in tqdm(os.listdir(args.data_dir)):
        fo = h5py.File(args.data_dir+"/"+dataset, "r")
        root_key = list(fo)[0]
        drug_list = list(fo[root_key].keys())

        output_df = pd.DataFrame()
        output_df["drugID"] = drug_list
        output_df["active"] = output_df["drugID"].apply(lambda x: int("active" in x))
        output_df["receptor"] = [root_key] * output_df.shape[0]

        output_df.to_csv(args.list_dir+"/"+str(root_key)+".csv", index=False)
        fo.close()
    print("finished generating lists.")
