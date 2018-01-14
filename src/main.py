import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from mol_graph import MolGraph, graph_from_mol, degrees




# may not need array_rep_from_mol
def array_rep_from_mol(mol):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_mol(mol)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        # arrayrep[('atom_neighbors', degree)] = \
        #     np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

data = np.ndarray(0)
inf = open('/mounts/u-spa-d2/grad/derek/Downloads/abl1/actives_final.sdf','rb')
fsuppl = Chem.ForwardSDMolSupplier(inf)
for mol in fsuppl:
    if mol is None:
        continue
    print(mol.GetNumAtoms())
    # data = np.vstack(rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    # data = graph_from_mol(mol)
    data = array_rep_from_mol(mol)