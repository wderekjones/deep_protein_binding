import os
import argparse
import numpy as np
np.random.seed(0)
from scipy.stats import randint
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, help="number of jobs to submit", default=1)
parser.add_argument("-p", type=str, help="partition to submit jobs", default="debug")
parser.add_argument("-s", type=str, help="path to python script",
                    default="/scratch/wdjo224/deep_protein_binding/src/train.py")
parser.add_argument("--exp_name", type=str, help="base name of the experiments", default="binding_debug")

args = parser.parse_args()

process_dist = randint(1, 8)
readout_dist = randint(50, 149)
T_dist = randint(1, 4)
batch_dist = randint(50, 501)
lr_dist = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
slurm_queue = args.p
script = args.s

for job in tqdm(range(args.N)):
    n_train_process = process_dist.rvs()
    readout_dim = readout_dist.rvs()
    T = T_dist.rvs()
    batch_size = batch_dist.rvs()
    lr = np.random.choice(lr_dist)
    sbatch_env_setup = "source activate deep_protein_binding ; " \
                       "export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24;"
    sbatch_base = "sbatch -p {} -N 1 --wrap '".format(args.p) + sbatch_env_setup + " python {}".format(args.s)
    sbatch_param_sample = " --readout_dim={} --T={} --n_train_process={} --batch_size={} --lr={}".format(
        readout_dim, T, n_train_process, batch_size, lr)
    os.system(sbatch_base + sbatch_param_sample +
              " --exp_name=" + args.exp_name+"_$(date +%T)_" + str(job) + " --loss=bce --output_type=class --n_workers=1 --output_dim=2 --target=label --n_epochs=5 -D=/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv -c=/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv --train_idxs=/scratch/wdjo224/deep_protein_binding/src/train.npy --val_idxs=/scratch/wdjo224/deep_protein_binding/src/val.npy'")
