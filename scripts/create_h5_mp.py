'''
The purpose of this script is to take a csv file containing protein_binding data and create an h5 with a specified proportion
of examples to hold as a testing set.
by: Derek Jones
'''


def save_to_hdf5(kinase_name, output_name, feature_names):
    print("Creating output datset {}".format(output_name))
    output_file = h5py.File(output_name, "a", libver="latest")
    drug_names = kinase_data["drugID"].unique()
    output_file.create_group(kinase_name)


    for idx, drug_name in enumerate(drug_names):
        output_file[kinase_name].create_group(drug_name)
        for feature in feature_names:

            if feature in ["receptor", "drugID", "smiles"]:
                output_file[kinase_name][drug_name].create_dataset(feature, [1],
                            dtype=h5py.special_dtype(vlen=str), data=kinase_data[kinase_data["drugID"] == drug_name][feature])

            else:
                output_file[kinase_name][drug_name][feature] = kinase_data[kinase_data["drugID"] == drug_name][feature]

    print("closing HDF5 file...")
    output_file.close()


if __name__ == "__main__":

    import time
    import h5py
    import pandas as pd
    import numpy as np
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
        save_to_hdf5(kinase_name=kinase_name, output_name=args.o+"_"+str(kinase_name)+"_"+str(time.time())+".h5",
                     feature_names=feature_names)
    t1 = time.clock()
    print(args.i, " with kinases {} converted to .h5 in {} seconds.".format(kinase_names, (t1-t0)))


