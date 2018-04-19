import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, help="name of the experiment to clean, don't specify if cleaning all",
                    default=None)
args = parser.parse_args()

print("starting to clean...")

for root, dirs, files in tqdm(os.walk("/scratch/wdjo224/deep_protein_binding/experiments")):
    for file in files:
        if args.exp_name is not None:
            if args.exp_name in root and "test_results" in file:
                os.system("rm {}".format(root+"/"+file))
        else:
            if "test_results" in file:
                os.system("rm {}".format(root+"/"+file))

print("done cleaning.")
