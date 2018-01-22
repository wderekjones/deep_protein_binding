'''
The purpose of this script is to take a csv file containing protein_binding data and create an h5 with a specified proportion
of examples to hold as a testing set.
by: Derek Jones
'''
import time
import os
import h5py
import pandas as pd
import numpy as np
import argparse
import numpy as np
from tqdm import tqdm
random_state = np.random.RandomState(0)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="path to old csv file")
parser.add_argument("-o", type=str, help="path to new h5 file")
args = parser.parse_args()


def save_to_hdf5(data_frame, output_name):

    output_file = h5py.File(output_name, "w", libver='latest')
    feature_names = list(data_frame.columns.values)
    kinase_names = list(set(data_frame['receptor']))

    for kinase_name in tqdm(kinase_names, total=len(kinase_names)):
        kinase_data = data_frame[data_frame['receptor'] == kinase_name]
        output_file.create_group(kinase_name)
        for drug in kinase_data["drugID"]:
            output_file[kinase_name].create_group(drug)
            compound_data = kinase_data[kinase_data["drugID"] == drug]
            for feature in iter(feature_names):
                if feature == "smiles":
                    output_file[kinase_name][drug].create_dataset(feature, [len(str(compound_data[feature].values))],
                                                                  dtype=h5py.special_dtype(vlen=str))
                elif feature not in ['receptor', 'drugID']:
                    output_file[kinase_name][drug][feature] = np.asarray(pd.to_numeric(compound_data[feature]), dtype=np.float16)
                else:
                    continue

    output_file.close()




t0 = time.clock()
save_to_hdf5(pd.read_csv(args.i, keep_default_na=False, na_values=[np.nan, 'na']), args.o)
t1 = time.clock()
print(args.i, "converted to .h5 in", (t1-t0), "seconds.")


