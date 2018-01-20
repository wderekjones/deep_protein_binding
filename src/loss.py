import torch
import torch.nn as nn
from collections import OrderedDict

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()
        self._forward_pre_hooks = OrderedDict()

    def forward(self, pred, target):
        return torch.mean(torch.pow((pred - target),2))
