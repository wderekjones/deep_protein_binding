import sys
sys.path.append("/scratch/wdjo224/deep_protein_binding")
import torch
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from torch.autograd import Variable
import src.dc_features as dc
from src.MoleculeDataset import MoleculeDatasetCSV
from src.utils import get_parser

args = get_parser().parse_args()


molecules = MoleculeDatasetCSV(csv_file=args.D, corrupt_path=args.c, target="label",
                                   scaling=args.scale).data
# pbar = tqdm(total=len(molecules))


def construct_multigraph(smiles):
    g = {}
    h = {}
    molecule = Chem.MolFromSmiles(smiles)
    for i in range(0, molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)

        h[i] = Variable(
            torch.from_numpy(dc.atom_features(atom_i, explicit_H=True)).view(1, 70)).float()

        for j in range(0, molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                # e_ij = map(lambda x: 1 if x == True else 0,
                #             dc.feat.graph_features.bond_features(e_ij))  # ADDED edge feat
                e_ij = map(lambda x: 1 if x == True else 0,
                           dc.bond_features(e_ij))

                e_ij = Variable(torch.from_numpy(np.fromiter(e_ij, dtype=float))).view(1, 6).float()
                atom_j = molecule.GetAtomWithIdx(j)
                if i not in g:
                    g[i] = []
                    g[i].append((e_ij, j))

    return {"g": g, "h": h}


def process_data(idx):
    # pbar.update(1)
    print("got the smiles")
    smile = molecules.iloc[idx].smiles
    return construct_multigraph(smile)


def main():
    import torch.multiprocessing as mp
    # mp.set_sharing_strategy("file_system")

    print("spawning the pool..")
    pool = mp.Pool(1)

    print("computing the graphs..")
    result = pool.map(process_data, (list(range(len(molecules)))))

    print("closing pool...")
    pool.close()
    print("joining pool...")
    pool.join()
    print("pool joined.")


if __name__ == "__main__":
    main()
