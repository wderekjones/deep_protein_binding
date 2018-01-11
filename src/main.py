import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops


data = np.ndarray(0)
inf     = open('/home/derek/Downloads/actives_final.sdf','rb')
fsuppl = Chem.ForwardSDMolSupplier(inf)
for mol in fsuppl:
    if mol is None:
        continue
    print(mol.GetNumAtoms())
    data = np.vstack(rdmolops.GetAdjacencyMatrix(mol))