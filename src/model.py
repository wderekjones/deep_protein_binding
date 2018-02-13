import torch
from torch import nn

#TODO: apply intitialization methods for the mpnn network
#TODO: try another network, maybe one that operates on morgan fingerprints? ECFP?


class MPNN(nn.Module):

    def __init__(self, T=3, p=0.5, n_tasks=1, n_atom_feats=70, n_bond_feats=146, name="mpnn"):
        super(MPNN, self).__init__()
        self.T = T
        self.name = name
        self.n_tasks = n_tasks
        self.n_atom_feats = n_atom_feats
        self.n_bond_feats = n_bond_feats
        self.R = nn.Linear(140, 128)
        self.U = nn.ModuleList([nn.Linear(self.n_bond_feats, self.n_atom_feats)] * self.T)
        self.V = nn.ModuleList([nn.Linear(self.n_atom_feats, self.n_atom_feats)] * self.T)
        self.E = nn.Linear(6, 6)
        self.dropout = nn.Dropout(p=p)
        self.shared_hidden1 = nn.Linear(128,128)
        self.shared_hidden2 = nn.Linear(128,100)
        self.hidden_layer_list = nn.ModuleList([nn.Linear(100, 100)]*self.n_tasks)
        self.output_layer_list = nn.ModuleList([nn.Linear(100, 1)]*self.n_tasks)  # create a list of output layers based on number of tasks

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight.data)  # pretty sure that this is the default pytorch initialization

    def init_weights(self):
        self.apply(self.weights_init)

    def get_n_hidden_units(self):
        n_units = 0
        for child in self.children():
            for param in child.named_parameters():
                if 'weight' in param[0]:
                    n_units += param[1].data.size()[0]
                # elif 'bias' in param[0]
        return n_units


    def readout(self, h, h2):
        catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], dim=1), zip(h2.keys(), h.keys()))
        outputs = []
        feature_map = torch.sum(torch.cat(list(map(lambda x: nn.ReLU()(self.R(x)), catted_reads)),dim=0),dim=0)
        for output_layer, hidden_layer in zip(self.output_layer_list,self.hidden_layer_list):
            # outputs.append(output_layer(torch.nn.Tanh()(hidden_layer(self.dropout(feature_map)))))
            outputs.append(output_layer(torch.nn.ReLU()(hidden_layer(self.shared_hidden2(torch.nn.ReLU()(torch.nn.ReLU()(self.shared_hidden1(feature_map))))))))
        return outputs

    def message_pass(self,g,h,k):
        # for each node v in the graph G
        for v in g.keys():
            neighbors = g[v]
            # for each neighbor of v
            for neighbor in neighbors:
                e_vw = neighbor[0]  # get the edge feature variable
                w = neighbor[1]  # get the bond? feature variable?

                m_w = self.V[k](h[w])  # compute the message vector
                m_e_vw = self.E(e_vw)
                reshaped = torch.cat((h[v], m_w, m_e_vw), 1)
                h[v] = nn.ReLU()(self.U[k](reshaped))

    def forward(self, h, g):
        h2 = h
        g2 = g

        for k in range(0, self.T):
            self.message_pass(h=h, g=g, k=k)
        x = self.readout(h=h, h2=h2)
        return x


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




