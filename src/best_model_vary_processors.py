import sys
sys.path.append("/scratch/wdjo224/deep_protein_binding")
import os
import numpy as np
np.random.seed(0)
import pandas as pd
from tqdm import tqdm


best_model_args = pd.read_csv("/scratch/wdjo224/deep_protein_binding/experiments/binding_debug_22:56:36_34/args.csv", index_col=0).drop(0)


for i in tqdm(range(12, 0, -1)):
    sbatch_env_setup = "source activate deep_protein_binding ; " \
                       "export LD_PRELOAD=/home/wdjo224/anaconda3/lib/libstdc++.so.6.0.24;"
    sbatch_base = "sbatch -p GPU2 --wrap '" + sbatch_env_setup + \
                  " python /scratch/wdjo224/deep_protein_binding/src/train.py"
    sbatch_param_sample = "--n_train_process={}".format(i)
    batch_args = str(" ".join(best_model_args["0"].values.tolist())) + " --exp_name=best_model_search_final" + \
    "_$(date +%T)_{} ".format(i) + sbatch_param_sample + " --n_workers=0'"
    os.system(sbatch_base+" "+batch_args)
