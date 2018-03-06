import pandas as pd
import os
import argparse

#TODO: specify loss or r2 or both (make subdirs?)

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="path to result directory", default="/home/derek/workspace/deep_protein_binding/results")
args = parser.parse_args()


if __name__ == "__main__":

    for root, _,  filelist in os.walk(args.i):
        for file in filelist:
            title = file.split("_")[0] + "_" +file.split("_")[1]+ "_" + file.split("_")[2].split(".")[0]#infer the chart name based on
            if not os.path.exists("/home/derek/workspace/deep_protein_binding/results/summary/"):
                os.makedirs("/home/derek/workspace/deep_protein_binding/results/summary/")
            pd.read_csv(root+"/"+file).describe().to_csv(root+"/summary/"+title+"_summary.csv")


