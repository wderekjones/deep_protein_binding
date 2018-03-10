# TODO: get some distribution objects from scipy(?) and then submit up to N sbatch jobs filling in the arguments
# just use uniform distribtuions

import os
import argparse
from scipy.stats import uniform, randint
from src.utils import get_args

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, help="number of jobs to submit")
args = parser.add_argument()

train_args = get_args()


process_dist = randint(1, 8)
readout_dist = randint(50, 200)


