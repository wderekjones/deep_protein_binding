import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


# TODO: test this loss function
# TODO: output weights for model inference mode
# takes a scaling prior, from which to automatically optimize for a model that weights task uncertainity
class MultiTaskHomoscedasticLoss(nn.Module):
    def __init__(self, n_tasks, prior=None):
        super(MultiTaskHomoscedasticLoss, self).__init__()
        self.n_tasks = n_tasks
        self.loss_list = nn.ModuleList([nn.MSELoss()]*self.n_tasks)

        if prior == "unweighted":
            self.loss_weights = Variable(torch.ones(self.n_tasks), requires_grad=True)
        elif prior == "uniform":
            self.loss_weights = Variable(torch.rand(self.n_tasks), requires_grad=True)
        elif prior == "normal":
            self.loss_weights = Variable(torch.normal(mean=0.5,std=torch.ones(self.n_tasks)), requires_grad=True)
        else:
            raise Exception("Not a valid prior..")

    def forward(self, batch_dict):
        loss = Variable(torch.zeros(1).float(), requires_grad=False)

        for idx, target in enumerate(batch_dict.keys()):
            target_loss = (1/(2 * (self.loss_weights[idx] * self.loss_weights[idx]))) * \
                   self.loss_list[idx](torch.stack(batch_dict[target]["pred"]),
                                       torch.stack(batch_dict[target]["true"])) + \
                   torch.log(self.loss_weights[idx] * self.loss_weights[idx])
            batch_dict[target]["loss"] = target_loss.data.cpu().numpy().ravel()
            batch_dict[target]["loss_weight"] = self.loss_weights[idx].data.cpu().numpy()
            loss += target_loss
        return loss


# TODO: test this loss function
# takes a scaling prior, with the assumption all tasks are equally weighted, and applies to loss functions
class MultiTaskWeightedLoss(nn.Module):
    def __init__(self, n_tasks, prior=None):
        super(MultiTaskWeightedLoss, self).__init__()
        self.n_tasks = n_tasks
        self.loss_list = nn.ModuleList([nn.MSELoss()] * self.n_tasks)
        if prior is not None:
            self.loss_weights = Parameter(torch.from_numpy(prior).float()) # could I not just use a numpy array
        else:
            self.loss_weights = Parameter(torch.ones(self.n_tasks))

    def forward(self, batch_dict):
        loss = Variable(torch.zeros(1).float(), requires_grad=False)

        for idx, target in enumerate(batch_dict.keys()):

            target_loss = self.loss_weights[idx] * self.loss_list[idx](torch.stack(batch_dict[target]["pred"]),
                                    torch.stack(batch_dict[target]["true"]))
            batch_dict[target]["loss"] = target_loss.data.cpu().numpy().ravel()
            batch_dict[target]["loss_weight"] = self.loss_weights[idx].data.cpu().numpy()
            loss += target_loss

        return loss


# TODO: test this loss function
# automatically optimizes for n normal distribution over loss weight values
class MultiTaskNormalLoss(nn.Module):
    def __init__(self, n_tasks):
        super(MultiTaskNormalLoss, self).__init__()
        self.n_tasks = n_tasks
        self.loss_list =nn.ModuleList([nn.MSELoss()]*self.n_tasks)
#       # use a normal over n_dimensional vectors with the assumption that the individual weights are not independent
        self.mean = Variable(torch.zeros(self.n_tasks), requires_grad=True)
        self.std = Variable(torch.ones(self.n_tasks), requires_grad=True)

    def forward(self, batch_dict):
        loss = Variable(torch.zeros(1).float(), requires_grad=False)

        # parameterize the distribution, then sample to get updated weight values
        loss_weights = torch.distributions.Normal(self.mean, self.std).sample()
        for idx, target in enumerate(batch_dict.keys()):
            target_loss = loss_weights[idx] * self.loss_list[idx](torch.stack(batch_dict[target]["pred"]),
                                                                    torch.stack(batch_dict[target]["true"]))

            batch_dict[target]["loss"] = target_loss.data.cpu().numpy().ravel()
            batch_dict[target]["loss_weight"] = loss_weights[idx].data.cpu().numpy()
            loss += target_loss

        return loss
