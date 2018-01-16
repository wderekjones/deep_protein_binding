import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help="path to directory containing data")
parser.add_argument('-o', type=str, help="output path", default="test.csv")
args = parser.parse_args()

print("reading data.")
output_dataframe = pd.DataFrame()
for dir in tqdm(iter(os.listdir(args.i))):
    if dir != "src":
        for content in os.listdir(args.i+'/'+dir):
            if ".ism" in content and "final" in content:
                content_path = args.i+'/'+dir+'/'+content
                content_df = pd.read_csv(content_path, sep=" ", header=None)
                if "actives" in content:
                    content_df["active"] = np.ones([content_df.shape[0]])
                elif "decoys" in content:
                    content_df["active"] = np.zeros([content_df.shape[0]])
                # print(content_df.columns)
                output_dataframe = pd.concat([content_df,output_dataframe])

# output_dataframe[[0, "active"]].columns = ["smiles", "active"]
output_dataframe.rename(columns={0:"smiles"}, inplace=True)
output_dataframe[["smiles", "active"]].to_csv(args.o, index=False)

print("output file generated.")