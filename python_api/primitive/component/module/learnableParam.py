import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleVariable(nn.Module):
    def __init__(self, init_val) -> None:
        super(SingleVariable, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, shape):
        return torch.ones(shape) * torch.exp(self.variance * 10.0)