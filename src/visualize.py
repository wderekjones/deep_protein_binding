import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to input directory containing results")
    args = parser.parse_args()

    return args


def main():

    args = get_args()

    for root, _,  filelist in os.walk(args.i):
        for file in filelist:
            data = pd.read_csv(root+"/"+file)
            title = file.split("_")[0].capitalize() + " " + file.split("_")[1].capitalize() + " " + file.split("_")[2].split(".")[0]#infer the chart name based on this
            plt.clf()
            plt.Figure(figsize=(10,10))
            plt.title(title)
            sns.distplot(data, color="g", kde_kws={"shade": True})
            plt.savefig(root+"/figs/"+title.replace(" ", "_"))


if __name__ == "__main__":

    if not os.path.exists("/home/derek/workspace/deep_protein_binding/results/figs/"):
        os.makedirs("/home/derek/workspace/deep_protein_binding/results/figs/")

    main()