'''
    Iterate over the experiment directory, run the testing script for each model, collect the best model
    wrt test performance
'''
import sys
sys.path.append("/scratch/wdjo224/deep_protein_binding")
import os
from tqdm import tqdm
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", help="path to experiment directory",
                    default="/scratch/wdjo224/deep_protein_binding/experiments")
parser.add_argument("--partition", type=str, help="partition to submit jobs", default="debug")
parser.add_argument("--n_test_process", type=int, help="number of testing processes to use", default=1)
parser.add_argument("--epoch", type=int, help="epoch from which to load weights", default=4)
# TODO: rename the script, evaluate experiments or something of that nature
# TODO: add argument for which epoch to use
args = parser.parse_args()

for dir in tqdm(os.listdir(args.exp_dir)):
    model = None
    for item in os.listdir(args.exp_dir + "/" + dir):
        if item == "args.csv":
            arg_df = pd.read_csv(args.exp_dir + "/" + dir + "/args.csv", index_col=0).drop(0, axis=0)
            args_string = " ".join(tuple([x for x in arg_df['0']]))
            for chkpnt in os.listdir(args.exp_dir + "/" + dir + "/checkpoints"):
                if "epoch{}".format(args.epoch) in chkpnt:
                    pid = "_".join([chkpnt.split("_")[0], chkpnt.split("_")[1]])
                    model_path = args.exp_dir + "/" + dir + "/checkpoints" + "/" + chkpnt
                    os.system("sbatch -p {} --wrap 'source activate deep_protein_binding; export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24; python test.py {} --model_path={} --pid={} --n_test_process={} --epoch={}'".format(args.partition, args_string, model_path, pid, args.n_test_process, args.epoch))

