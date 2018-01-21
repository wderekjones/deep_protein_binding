import torch
from torch import nn


class LinearNetwork(nn.Module):

    def __init__(self, batch_size, atom_0, atom_1, atom_2, bond_0, bond_1, bond_2, bond_3, edge_0, edge_1, edge_2,
                 hidden_00, hidden_01, hidden_02, hidden_10, num_outputs):
        super(LinearNetwork, self).__init__()

        # store the dimension paramters
        # remove the 0 indexed shapes as these are equivalent to batch_size...
        self.batch_size = batch_size
        self.atom_0 = atom_0
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.bond_0 = bond_0
        self.bond_1 = bond_1
        self.bond_2 = bond_2
        self.bond_3 = bond_3
        self.edge_0 = edge_0
        self.edge_1 = edge_1
        self.edge_2 = edge_2
        self.hidden_00 = hidden_00
        self.hidden_01 = hidden_01
        self.hidden_02 = hidden_02
        self.hidden_10 = hidden_10
        self.num_outputs = num_outputs
        self.atom_relu0 = nn.ReLU()
        self.bond_relu0 = nn.ReLU()
        self.edge_relu0 = nn.ReLU()
        self.atom_relu1 = nn.ReLU()
        self.bond_relu1 = nn.ReLU()
        self.edge_relu1 = nn.ReLU()
        self.combined_relu = nn.ReLU()

        # create the fully connected layers
        self.atom_fc0 = nn.Linear(self.atom_2, hidden_00)
        self.bond_fc0 = nn.Linear(self.bond_3, hidden_01)
        self.edge_fc0 = nn.Linear(self.edge_2, hidden_02)
        self.atom_fc1 = nn.Linear((self.atom_0*self.atom_1*self.hidden_00),self.hidden_10)
        self.bond_fc1 = nn.Linear((self.bond_0*self.bond_1*self.bond_2*self.hidden_01), self.hidden_10)
        self.edge_fc1 = nn.Linear((self.edge_0 * self.edge_1 * self.hidden_02), self.hidden_10)
        self.output_fc = nn.Linear(self.hidden_10, num_outputs)

    def forward(self, atom_input, bond_input, edge_input):

        atom_output0 = self.atom_relu0(self.atom_fc0(atom_input))
        bond_output0 = self.bond_relu0(self.bond_fc0(bond_input))
        edge_output0 = self.edge_relu0(self.edge_fc0(edge_input))

        atom_output1 = self.atom_relu1(self.atom_fc1(atom_output0.view(-1)))
        bond_output1 = self.bond_relu1(self.bond_fc1(bond_output0.view(-1)))
        edge_output1 = self.edge_relu1(self.edge_fc1(edge_output0.view(-1)))

        combined = torch.add(atom_output1,bond_output1)
        combined = torch.add(edge_output1,combined)
        combined = combined.view(self.batch_size,self.num_outputs)
        # return self.combined_relu(self.output_fc(combined))
        return self.combined_relu(combined)
