from multiprocessing import cpu_count
from MoleculeDataset import MoleculeDataset
from torch.utils.data import DataLoader


molecules = MoleculeDataset("/u/vul-d1/scratch/wdjo224/deep_protein_binding/test.csv")
molecules.read_csv_data()
molecules.clean_csv_data(workers=cpu_count()-1)
molecules.tensorize_smiles()

molecule_loader = DataLoader(molecules, batch_size=32, shuffle=True, num_workers=cpu_count()-1)

for i_batch, sample_batched in enumerate(molecule_loader):
    print("now loading batch {}".format(i_batch))
