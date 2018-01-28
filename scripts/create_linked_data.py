import h5py
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="path to input directory containing the files")
parser.add_argument("-o", type=str, help="output path")
args = parser.parse_args()

if __name__ == "__main__":
    data_dir = args.i
    output_path = args.o
    fo = h5py.File(args.o, "w")

    for file in os.listdir(data_dir):
        fo_path = data_dir + "/" + file
        fo_name = file.split("_")[5]  # this is the expected position of the kinase name
        fo[fo_name] = h5py.ExternalLink(fo_path, fo_name)
    fo.close()
