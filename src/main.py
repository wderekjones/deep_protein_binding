import torch
import numpy as np
from multiprocessing import cpu_count
from MoleculeDataset import MoleculeDataset
from torch.utils.data import DataLoader
from models import LinearNetwork

# molecules = MoleculeDataset("/u/vul-d1/scratch/wdjo224/deep_protein_binding/test.csv")
molecules = MoleculeDataset("/media/derek/Data/thesis_data/kinase_dragon_smiles.csv", ["MW","AMW"], nrows=10000, num_workers=cpu_count()-2)
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
output_dim = 2
model = LinearNetwork(batch_size=batch_size, atom_0=batch_size, atom_1=atom_shape[0], atom_2=atom_shape[1],
                      bond_0=batch_size, bond_1=bond_shape[0], bond_2=bond_shape[1], bond_3=bond_shape[2],
                      edge_0=batch_size, edge_1=edge_shape[0], edge_2=edge_shape[1],
                      hidden_00=10, hidden_01=10, hidden_02=10, hidden_10=batch_size*output_dim, num_outputs=output_dim)

# model.cuda()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

for epoch in range(num_epochs):
    epoch_losses = []
    optimizer.zero_grad()
    for i_batch, sample_batched in enumerate(molecule_loader):
        # print("now loading batch {}".format(i_batch))
        pred = model(torch.autograd.Variable(sample_batched['atom'].float()),
          torch.autograd.Variable(sample_batched['bond'].float()),
          torch.autograd.Variable(sample_batched['edge'].float()))

        loss = loss_fn(pred,torch.autograd.Variable(sample_batched['target'].float()))
        loss.backward()
        epoch_losses.append(loss.data.numpy())
    optimizer.step()
    print("epoch: {} \t loss: {} ".format(epoch, np.mean(epoch_losses)))
