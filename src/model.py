import torch
import torch.functional as F
import numpy as np
import deepchem as dc
from torch import nn
from torch.autograd import Variable
from collections import OrderedDict
from rdkit import Chem


class MPNN(nn.Module):

    def __init__(self, T):
        super(MPNN, self).__init__()
        self.T = T
        self.R = nn.Linear(150, 128)
        self.U = {0: nn.Linear(156, 75), 1: nn.Linear(156, 75), 2: nn.Linear(156, 75)}
        self.V = {0: nn.Linear(75, 75), 1: nn.Linear(75, 75), 2: nn.Linear(75, 75)}
        self.E = nn.Linear(6, 6)
        self.output = nn.Linear(128,1)

    def readout(self, h, h2):
        catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], 1), zip(h2.keys(), h.keys()))
        activated_reads = map(lambda x: nn.ReLU()(self.R(x)), catted_reads)
        readout = Variable(torch.zeros(1, 128))
        for read in activated_reads:
            readout = readout + read
        return F.tanh(readout)

    def message_pass(self,h,g,k):
        for v in g.keys():
            neighbors = g[v]
            for neighbor in neighbors:
                e_vw = neighbor[0]  # feature variable
                w = neighbor[1]

                m_w = self.V[k](h[w])
                m_e_vw = self.E(e_vw)
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1)
                h[v] = nn.ReLU()(self.U[k](reshaped))


    def forward(self, h, g):
        h2 = h
        g2 = g

        for k in range(0, self.T):
            self.message_pass(g,h,k)
        x = self.readout(h, h2)
        return self.output(x)



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




