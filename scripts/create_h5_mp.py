'''
The purpose of this script is to take a csv file containing protein_binding data and create an h5 with a specified proportion
of examples to hold as a testing set.
by: Derek Jones
'''



def compute_drug_data(drug_data, drug_name, feature_names):

    data_dict = {}
    data_dict["drugID"] = drug_name
    for feature in iter(feature_names):
        data_dict[feature] = drug_data[feature].values[0]
    return data_dict


def write_drug_data(data_dict, kinase_name, output_file):
    output_file[kinase_name].create_group(data_dict["drugID"])
    for feature in data_dict.keys():
        if feature in ["receptor", "drugID", "smiles"]:
            output_file[kinase_name][data_dict["drugID"]].create_dataset(feature, [1],
                                                                         dtype=h5py.special_dtype(vlen=str),
                                                                         data=data_dict[feature])
        else:
            output_file[kinase_name][data_dict["drugID"]][feature] = data_dict[feature]


def save_to_hdf5(kinase_name, kinase_data, output_name, feature_names, num_processes):
    print("Creating {}-process pool".format(num_processes))
    pool = mp.Pool(num_processes)

    print("Creating output datset {}".format(output_name))
    output_file = h5py.File(output_name, "a", libver="latest")
    drug_names = kinase_data["drugID"].unique()
    output_file.create_group(kinase_name)

    result = pool.starmap(compute_drug_data, [(kinase_data[kinase_data["drugID"] == drug_name],
                                               drug_name, feature_names) for drug_name in drug_names])


    for idx, data_dict in enumerate(result):
        output_file[kinase_name].create_group(data_dict["drugID"])
        for feature in data_dict.keys():

            if feature in ["receptor", "drugID", "smiles"]:
                output_file[kinase_name][data_dict["drugID"]].create_dataset(feature, [1],
                            dtype=h5py.special_dtype(vlen=str), data=data_dict[feature])

            else:
                output_file[kinase_name][data_dict["drugID"]][feature] = data_dict[feature]

    print("Shutting down process pool")
    pool.close()
    pool.join()

    print("closing HDF5 file...")
    output_file.close()



if __name__ == "__main__":
    import time
    import os
    import h5py
    import pandas as pd
    import numpy as np
    import multiprocessing as mp
    import argparse
    import numpy as np
    from tqdm import tqdm

    random_state = np.random.RandomState(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to old csv file")
    parser.add_argument("-k", type=str, nargs='+', help="kinases to include in job", default=None)
    parser.add_argument("-o", type=str, help="path to new h5 file")
    args = parser.parse_args()

    t0 = time.clock()
    print("reading csv file...")
    data_frame = pd.read_csv(args.i, keep_default_na=False, na_values=[np.nan, 'na'])
    print("csv file loaded into memory, now beginning to write h5 file...")
    feature_names = list(data_frame.columns.values)
    kinase_names = args.k
    if kinase_names is None: kinase_names = data_frame["receptor"].unique()
    print("using kinases: ")
    for name in kinase_names:
        print(name)
    for kinase_name in tqdm(kinase_names, total=len(kinase_names)):
        kinase_data = data_frame[data_frame['receptor'] == kinase_name]
        save_to_hdf5(kinase_name=kinase_name, kinase_data=kinase_data, output_name=args.o+"_"+str(time.time())+".h5",
                     feature_names=feature_names, num_processes=mp.cpu_count()-1)
    t1 = time.clock()
    print(args.i, " with kinases {} converted to .h5 in {} seconds.".format(kinase_names, (t1-t0)))


