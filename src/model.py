import time
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score
#TODO: try another network, maybe one that operates on morgan fingerprints? ECFP?


class Model(nn.Module):
    def __init__(self, target_list, output_type="regress", output_dim=1):
        super(Model, self).__init__()
        self.target_list = target_list
        self.output_type = output_type
        self.output_dim = output_dim


class MPNN(Model):

    def __init__(self, target_list, T=3, p=0.5, n_atom_feats=70, n_bond_feats=146, name="mpnn", output_type="regress",
                 output_dim=1):
        super(MPNN, self).__init__(target_list=target_list, output_type=output_type, output_dim=output_dim)
        self.T = T
        self.name = name
        self.n_tasks = len(target_list)
        self.target_list = target_list
        self.output_type = output_type
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
        self.output_layer_list = nn.ModuleList([nn.Linear(100, self.output_dim)]*self.n_tasks)  # create a list of output layers based on number of tasks

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
        hidden_outputs = self.compute_hidden(self.concat_reads(h, h2))
        outputs = []
        for output_layer, hidden_output in zip(self.output_layer_list, hidden_outputs):
            if self.output_type == "regress":
                outputs.append(output_layer(torch.nn.ReLU()(hidden_output)))
            elif self.output_type == "class":
                outputs.append(nn.Softmax()(output_layer(hidden_output)))

        return outputs

    def compute_hidden(self, feature_map):
        outputs = []
        shared_hidden = self.compute_shared_hidden(feature_map)
        for hidden_layer in self.hidden_layer_list:
            outputs.append(torch.nn.ReLU()(hidden_layer(shared_hidden)))
        return outputs

    def compute_shared_hidden(self, feature_map):
        return self.shared_hidden2(torch.nn.ReLU()(torch.nn.ReLU()(self.shared_hidden1(feature_map))))

    def concat_reads(self, h, h2):

        catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], dim=1), zip(h2.keys(), h.keys()))
        feature_map = torch.sum(torch.cat(list(map(lambda x: nn.ReLU()(self.R(x)), catted_reads)),dim=0),dim=0)
        return feature_map

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

    def forward(self, h, g, hidden_pass=False):
        if hidden_pass:
            return self.get_shared_hidden_features(h,g)
        h2 = h
        g2 = g

        for k in range(0, self.T):
            self.message_pass(h=h, g=g, k=k)
        x = self.readout(h=h, h2=h2)
        return x

    def get_shared_hidden_features(self, h, g):
        h2 = h

        for k in range(0, self.T):
            self.message_pass(h=h, g=g, k=k)
        x = self.compute_shared_hidden(self.concat_reads(h=h, h2=h2))
        return x

    def get_task_hidden_features(self, h, g, task_no):
        h2 = h

        for k in range(0, self.T):
            self.message_pass(h=h, g=g, k=k)
        x = self.compute_shared_hidden(self.concat_reads(h=h, h2=h2))
        x = self.compute_hidden(x)
        return x[task_no]

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

        if self.output_type == "regress":

            batch_dict = {key: {"batch_size": len(batch), "pred": [], "true": [], "loss": [], "metrics": {"r2": []}} for
                          key in
                          self.target_list}
        elif self.output_type == "class":
            batch_dict = {
            key: {"batch_size": len(batch), "pred": [], "true": [], "loss": [], "metrics": {"acc": [], "prec": [],
                                                                                            "rec": [], "f1": []}} for
            key in self.target_list}

        start_clock = time.clock()

        for data in tqdm(batch, total=len(batch)):

            # Forward pass: compute output of the network by passing x through the model. Get N outputs as a result
            y_pred = self.forward(h=data["h"], g=data["g"])

            for idx, (pred, target) in enumerate(zip(y_pred, self.target_list)):

                # check to see if model is on GPU, as suggested in https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
                if next(self.parameters()).is_cuda:
                    batch_dict[target]["true"].append(Variable(torch.from_numpy(data[target]).float().cuda()))
                    batch_dict[target]["pred"].append(pred)

                else:
                    batch_dict[target]["true"].append(Variable(torch.from_numpy(data[target])).float())
                    batch_dict[target]["pred"].append(pred)

        for target in self.target_list:
            if self.output_type == "class":
                y_true = Variable(torch.from_numpy(OneHotEncoder(n_values=self.output_dim, sparse=False).fit_transform(
                    torch.stack(batch_dict[target]["true"]).data.numpy()))) # convert target to n-class vector here
                batch_dict[target]["true"] = y_true
            else:
                y_true = Variable(torch.from_numpy(torch.stack(batch_dict[target]["true"]).data.numpy()))
                batch_dict[target]["true"] = y_true
            y_pred = torch.stack(batch_dict[target]["pred"]).data.numpy()
            if self.output_type == "regress":
                batch_dict[target]["metrics"]["r2"] = r2_score(y_true=y_true.data.numpy(),
                                                               y_pred=y_pred)
            elif self.output_type == "class":
                batch_dict[target]["metrics"]["acc"] = accuracy_score(y_true=np.argmax(y_true.data.numpy(),axis=1),
                                                                      y_pred=np.argmax(y_pred, axis=1))
                batch_dict[target]["metrics"]["prec"] = precision_score(y_true=np.argmax(y_true.data.numpy(),axis=1),
                                                                        y_pred=np.argmax(y_pred, axis=1))
                batch_dict[target]["metrics"]["rec"] = recall_score(y_true=np.argmax(y_true.data.numpy(),axis=1),
                                                                    y_pred=np.argmax(y_pred, axis=1))
                batch_dict[target]["metrics"]["f1"] = f1_score(y_true=np.argmax(y_true.data.numpy(), axis=1),
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




