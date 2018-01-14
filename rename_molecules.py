import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help="path to directory containing data")
args = parser.parse_args()

output_dataframe = pd.DataFrame()
for dir in iter(os.listdir(args.i)):
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

print(output_dataframe.shape)