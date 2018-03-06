'''
    This script implements methods in order to benchmark the performance of the neural fingerprinting methods for the
    task of predicting dragon features from the smiles input

'''


from sklearn.linear_model import LinearRegression
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler, Normalizer


def load_data(csv_file, targets=None, scaling=None, corrupt_path=None):

    cols = ["receptor", "drugID", "smiles", "label"] + targets
    data = pd.read_csv(csv_file, usecols=cols)
    corrupt_compound_df = pd.read_csv(corrupt_path)
    data = data[~data.drugID.isin(corrupt_compound_df.drugID)]
    if scaling == "std":
        print("standardizing data...")
        data[targets] = StandardScaler().fit_transform(data[targets])
    elif scaling == "norm":
        print("normalizing data...")
        data[targets] = Normalizer().fit_transform(data[targets])
    elif scaling is not None:
        raise Exception("preprocessing method not implemented.")


    mols = [Chem.MolFromSmiles(smile) for smile in data["smiles"]]

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol) for mol in mols]

    return fps

def parallelize(data, func, workers):
    data_split = np.array_split(data, workers)
    pool = Pool(processes=workers)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def train(x_train, y_train):
    model = LinearRegression()

if __name__ == "__main__":
    data = load_data(csv_file="/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv",
              corrupt_path="/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv", targets=["Hy"],
              scaling="std")
    print(data.shape)