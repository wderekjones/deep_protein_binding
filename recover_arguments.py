'''
    forgot to save the command line arguments in the training script...so using this to parse
    the slurm output in order to recover the arguments...

'''


import os
import pandas as pd
from tqdm import tqdm

for item in tqdm(os.listdir()):
    exp_name, params = None, None
    if "slurm" in item:
        file = open(item, "r")
        for line in file:
            if "experiment:" in line:
                exp_name = line.strip("experiment:").replace("\n","").replace(" ","")
            elif "run parameters:" in line:
                params = pd.DataFrame(eval(line.strip("run parameters:")))  # convert string rep of list to a list of strings

        # Save the params (in TBD format) under ~/deep_protein_binding/experiments/exp_name
        params.to_csv("/scratch/wdjo224/deep_protein_binding/experiments/"+exp_name+"/args.csv",
                      header=False, index=False)
