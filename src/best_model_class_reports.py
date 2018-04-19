import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report


best_model_df = pd.read_csv("/scratch/wdjo224/deep_protein_binding/best_model_search_max_f1_score_models.csv")
result_df = pd.DataFrame()

for path in tqdm(best_model_df.path):
        test_df = pd.DataFrame({"idx": [], "pred": [], "true": [], "loss": []})
        for file in os.listdir(path):
            # print("{}/{}".format(path, file))
            test_df = pd.concat([test_df, pd.read_csv("{}/{}".format(path, file), index_col=0)])
        y_true = test_df.true.apply(lambda x: np.argmax(np.fromstring(x.strip("[ ]"), sep=" ", dtype=np.float32)))
        y_pred = test_df.pred.apply(lambda x: np.argmax(np.fromstring(x.strip("[ ]"), sep=" ", dtype=np.float32)))
        result_df = pd.concat([result_df, pd.DataFrame({"path": path, "report": classification_report(
            y_pred=y_pred, y_true=y_true)}, index=[0])])

result_df.reset_index().to_csv("/scratch/wdjo224/deep_protein_binding/"
                               "best_model_search_max_f1_score_class_reports.csv")
