import pandas as pd
import numpy as np

def read_labels(label_path, exclude_path):

    label_df = pd.read_csv(label_path, header=None)
    exclude_df = pd.read_csv(exclude_path, header=None)
    return np.setdiff1d(label_df[0].values, exclude_df[0].values).tolist()

if __name__ == "__main__":
    read_labels("/media/derek/Data/thesis_data/drug_features_list.csv", "/media/derek/Data/thesis_data/null_column_list.csv")