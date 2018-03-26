import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, r2_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, help="path to directory containing test results",
                    default="/scratch/wdjo224/deep_protein_binding/experiments")
parser.add_argument("--exp_name", type=str, help="name of the experiment to collect results", default="binding_debug")
parser.add_argument("--exp_type", type=str, help="indicate regression (reg) or classification (class)",
                    default="class")
parser.add_argument("--exp_epoch", type=int, help="which epoch to get results for", default=4)
args = parser.parse_args()

test_dict = {"path": [], "score": []}
test_list = []

print("reading test results...")

for root, dirs, files in tqdm(os.walk(args.exp_dir), total=len(os.listdir(args.exp_dir))):
    if "test_results" in root and args.exp_name in root and "epoch{}".format(args.exp_epoch) in root:
        process = root.split("/")[-1].split("_")[0]
        test_df = pd.DataFrame({"idx": [], "pred": [], "true": [], "loss": []})
        for file in os.listdir(root):
            test_df = pd.concat([test_df, pd.read_csv(root + "/" + file, index_col=0)])
        score = None
        if args.exp_type == "class":
            score = f1_score(np.asarray(test_df.true.apply(lambda x: np.argmax(eval(x))).values),
                                 np.asarray(test_df.pred.apply(lambda x: np.argmax(eval(x))).values))
        elif args.exp_type == "reg":
            score = r2_score(np.asarray(test_df.true.apply(lambda x: eval(x)).values),
                                 np.asarray(test_df.pred.apply(lambda x: eval(x)).values))
        else:
            raise Exception("not a valid output type")
        test_list.append({"path": root, "score": score, "process": process})

print("finished reading. finding best result")

best_score = -9999999
best_idx = 0
for idx, test in tqdm(enumerate(test_list)):
    if test["score"] > best_score:
        best_score = test["score"]
        best_idx = idx

best_test = test_list[best_idx]
print("best test results:\n score: {} \t process: {} \t path: {}".format(best_test["score"], best_test["process"],
                                                                         best_test["path"]))
pd.DataFrame(test_list).sort_values(by="score", ascending=False).to_csv(
    "/scratch/wdjo224/deep_protein_binding/"+args.exp_name+"_test_results.csv")
