import time
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score

import dc_features as dc
from rdkit import Chem


#TODO: try another network, maybe one that operates on morgan fingerprints? ECFP?


class MPNN(nn.Module):

    def __init__(self, target, T=3, p=0.5, n_atom_feats=70, n_bond_feats=146, readout_dim=128, name="mpnn", output_type="regress",
                 output_dim=1):
        super(MPNN, self).__init__()
        self.T = T
        self.name = name
        self.output_type = output_type
        self.output_dim = output_dim
        self.readout_dim = readout_dim
        self.target = target
        self.n_atom_feats = n_atom_feats # make sure this is correct naming
        self.n_bond_feats = n_bond_feats # make sure this is correct naming
        self.R = nn.Linear(140, self.readout_dim)
        self.U = nn.ModuleList([nn.Linear(self.n_bond_feats, self.n_atom_feats)] * self.T)
        self.V = nn.ModuleList([nn.Linear(self.n_atom_feats, self.n_atom_feats)] * self.T)
        self.E = nn.Linear(6, 6)
        self.output_layer = nn.Linear(self.readout_dim, self.output_dim)
        self.dropout = nn.Dropout(p=p)

    def get_n_hidden_units(self):
        n_units = 0
        for child in self.children():
            for param in child.named_parameters():
                if 'weight' in param[0]:
                    n_units += param[1].data.size()[0]
                # elif 'bias' in param[0]
        return n_units

    def output(self, h, h2):
        hidden_output = self.readout(h, h2)
        if self.output_type == "regress":
            return self.output_layer(hidden_output) # use a ReLU here?
        elif self.output_type == "class":
            return nn.Softmax()(self.output_layer(hidden_output))

    def readout(self, h, h2):
        catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], dim=1), zip(h2.keys(), h.keys()))
        return torch.sum(torch.cat(list(map(lambda x: nn.ReLU()(self.R(x)), catted_reads)),dim=0),dim=0)

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

    def forward(self, smiles, hidden_pass=False):
        # if hidden_pass:
        #     return self.get_shared_hidden_features(h,g)
        mol = self.construct_multigraph(smiles)
        h = mol["h"]
        g = mol["g"]
        h2 = h
        g2 = g

        for k in range(0, self.T):
            self.message_pass(h=h, g=g, k=k)
        x = self.output(h=h, h2=h2)
        return x

    def construct_multigraph(self, smiles):
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

        return {"g":g, "h":h}

    def train_step(self, batch, loss_fn):
        # make sure model is in train mode
        self.train()

        batch_dict = self.step(batch=batch, loss_fn=loss_fn)

        # update model gradients()
        batch_dict["loss"].backward()

        return batch_dict

    def validation_step(self, batch, loss_fn):
        # make sure model is evaluation mode
        self.eval()

        batch_dict = self.step(batch=batch, loss_fn=loss_fn)

        # put the model back into training mode
        self.train()

        return batch_dict

    def step(self, batch, loss_fn):

        batch_dict = {}

        if self.output_type == "regress":

            batch_dict = {key: {"batch_size": len(batch), "pred": [], "true": [], "loss": [], "metrics": {"r2": []}} for
                          key in
                          [self.target]}
        elif self.output_type == "class":
            batch_dict = {
            key: {"batch_size": len(batch), "pred": [], "true": [], "loss": [], "metrics": {"acc": [], "prec": [],
                                                                                            "rec": [], "f1": []}} for
            key in [self.target]}

        start_clock = time.clock()

        for data in tqdm(batch, total=len(batch)):

            # Forward pass: compute output of the network by passing x through the model. Get N outputs as a result
            y_pred = self.forward(data["smiles"])

            batch_dict[self.target]["true"].append(Variable(data[self.target]))
            batch_dict[self.target]["pred"].append(y_pred)

        if self.output_type == "class":
            y_true = Variable(torch.from_numpy(OneHotEncoder(n_values=self.output_dim, sparse=False).fit_transform(
                torch.stack(batch_dict[self.target]["true"]).data.numpy())))  # convert target to n-class vector here
            batch_dict[self.target]["true"] = y_true
        else:
            y_true = Variable(torch.from_numpy(torch.stack(batch_dict[self.target]["true"]).data.numpy()))
            batch_dict[self.target]["true"] = y_true
        y_pred = torch.stack(batch_dict[self.target]["pred"]).data.numpy()
        if self.output_type == "regress":
            batch_dict[self.target]["metrics"]["r2"] = r2_score(y_true=y_true.data.numpy(),
                                                           y_pred=y_pred)
        elif self.output_type == "class":
            batch_dict[self.target]["metrics"]["acc"] = accuracy_score(y_true=np.argmax(y_true.data.numpy(), axis=1),
                                                                  y_pred=np.argmax(y_pred, axis=1))
            batch_dict[self.target]["metrics"]["prec"] = precision_score(y_true=np.argmax(y_true.data.numpy(), axis=1),
                                                                    y_pred=np.argmax(y_pred, axis=1))
            batch_dict[self.target]["metrics"]["rec"] = recall_score(y_true=np.argmax(y_true.data.numpy(), axis=1),
                                                                y_pred=np.argmax(y_pred, axis=1))
            batch_dict[self.target]["metrics"]["f1"] = f1_score(y_true=np.argmax(y_true.data.numpy(), axis=1),
                                                           y_pred=np.argmax(y_pred, axis=1))

        stop_clock = time.clock()

        loss = loss_fn(batch_dict)

        # return a dictionary objects containing the metrics
        return {"loss": loss, "batch_dict": batch_dict, "time": (stop_clock - start_clock)}


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




