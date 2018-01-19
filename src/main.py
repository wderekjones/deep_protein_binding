import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from MoleculeDataset import MoleculeDataset
from torch.utils.data import DataLoader
from model import LinearNetwork
from label import read_labels

parser = argparse.ArgumentParser()
parser.add_argument('-D', type=str, help="path to dataset")
parser.add_argument('-L', type=str, help="path to labels to use")

args = parser.parse_args()

if __name__ == "__main__":

    labels = read_labels("/media/derek/Data/thesis_data/drug_features_list.csv", "/media/derek/Data/thesis_data/null_column_list.csv")
    molecules = MoleculeDataset(args.D, labels, nrows=1000, num_workers=cpu_count()-2)
    molecules.read_csv_data()
    molecules.clean_csv_data()
    molecules.tensorize_smiles()
    batch_size = 10
    num_epochs = 50

    molecule_loader = DataLoader(molecules, batch_size=batch_size, shuffle=True, num_workers=cpu_count()-2) # look at source to understand impact of num_workers

    print("Building the network.")
    atom_shape = molecules[0]['atom'].shape
    bond_shape = molecules[0]['bond'].shape
    edge_shape = molecules[0]['edge'].shape
    output_dim = len(labels)
    model = LinearNetwork(batch_size=batch_size, atom_0=batch_size, atom_1=atom_shape[0], atom_2=atom_shape[1],
                      bond_0=batch_size, bond_1=bond_shape[0], bond_2=bond_shape[1], bond_3=bond_shape[2],
                      edge_0=batch_size, edge_1=edge_shape[0], edge_2=edge_shape[1],
                      hidden_00=10, hidden_01=10, hidden_02=10, hidden_10=batch_size*output_dim, num_outputs=output_dim)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    print("training model")
    for epoch in range(num_epochs):
        epoch_losses = []
        optimizer.zero_grad()
        for i_batch, sample_batched in enumerate(molecule_loader):
            pred = model(torch.autograd.Variable(sample_batched['atom'].type(torch.HalfTensor).cuda()),
            torch.autograd.Variable(sample_batched['bond'].type(torch.HalfTensor).cuda()),
            torch.autograd.Variable(sample_batched['edge'].type(torch.HalfTensor).cuda()))
            loss = loss_fn(pred.cpu().type(torch.HalfTensor), torch.autograd.Variable(sample_batched['target'].type(torch.HalfTensor)))
            loss.backward()
            epoch_losses.append(loss.data.cpu().numpy())
        optimizer.step()
        print("epoch: {} \t loss: {} ".format(epoch, np.mean(epoch_losses)))
