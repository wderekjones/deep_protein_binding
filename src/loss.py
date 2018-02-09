import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()
        self._forward_pre_hooks = OrderedDict()

    def forward(self, pred, target):
        return torch.mean(torch.pow((pred - target),2))


class MultiTaskLoss(nn.Module):
    # TODO: accept a prior distribution over loss weights
    # TODO: add an option to "freeze" weights
    def __init__(self, n_tasks):
        super(MultiTaskLoss, self).__init__()
        self.n_tasks = n_tasks
        self.loss_list = nn.ModuleList([nn.MSELoss()]*self.n_tasks)
        self.loss_weights = Variable(torch.rand(self.n_tasks), requires_grad=True)

    def forward(self, batch_dict):
        loss = Variable(torch.zeros(1).float(), requires_grad=False)

        for idx, target in enumerate(batch_dict.keys()):
            loss += (1/(2 * (self.loss_weights[idx] * self.loss_weights[idx]))) * \
                    self.loss_list[idx](torch.stack(batch_dict[target]["pred"]),
                                        torch.stack(batch_dict[target]["true"])) + \
                    torch.log(self.loss_weights[idx] * self.loss_weights[idx])
        return loss
