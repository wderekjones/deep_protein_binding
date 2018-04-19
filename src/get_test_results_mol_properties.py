import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, help="epoch to evaulate on", default=2)
args = parser.parse_args()

property_list = ["Hy", "MLOGP", "MLOGP2", "PDI", "SAacc", "SAdon", "SAtot", "Uc", "Ui",
                 "VvdwMG", "VvdwZAZ", "Vx", "TPSA"]

for property in property_list:
    os.system("python get_test_results.py --exp_name={} --exp_type=reg --exp_epoch={}".format(
        property, args.epoch))
