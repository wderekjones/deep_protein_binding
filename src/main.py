import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from mol_graph import MolGraph, graph_from_mol, degrees


# this is the function that
def load_task_data(name, train_slices, test_slices):
    dataset_info = datasets_info[name]
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')
    full_data_path = os.path.join(data_dir, dataset_info['data_file'])

    train_data, test_data = load_data_slices(
        full_data_path,
        [[slice(*bounds) for bounds in train_slices],
         [slice(*bounds) for bounds in test_slices ]],
        input_name='smiles',
        target_name=dataset_info['target_name'])

    return train_data, test_data, dataset_info['nll_func']

# may not need array_rep_from_mol
# def array_rep_from_mol(mol):
#     """Precompute everything we need from MolGraph so that we can free the memory asap."""
#     molgraph = graph_from_mol(mol)
#     arrayrep = {'atom_features' : molgraph.feature_array('atom'),
#                 'bond_features' : molgraph.feature_array('bond'),
#                 'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
#                 'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
#     for degree in degrees:
#         # arrayrep[('atom_neighbors', degree)] = \
#         #     np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
#         arrayrep[('bond_neighbors', degree)] = \
#             np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
#     return arrayrep

data = np.ndarray(0)
inf     = open('/mounts/u-spa-d2/grad/derek/Downloads/abl1/actives_final.sdf','rb')
fsuppl = Chem.ForwardSDMolSupplier(inf)
for mol in fsuppl:
    if mol is None:
        continue
    print(mol.GetNumAtoms())
    # data = np.vstack(rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    data = graph_from_mol(mol)
