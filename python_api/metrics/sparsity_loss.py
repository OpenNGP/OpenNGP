import torch
import torch.nn as nn

from typing import List
from python_api.renderer.renderpass import RenderPassResult


class SparsityLoss(nn.Module):
    def __init__(self, weight, length) -> None:
        super(SparsityLoss, self).__init__()
        self.weight = weight
        self.length = length

    def forward(self, render_rets: List[RenderPassResult], batch):
        ret = [r for r in render_rets if 'sparsity_sample' == r.name]
        if 0 == len(ret):
            return {'sparsity': 0}
        sigmas = ret[0].sigmas
        loss = self.weight * (1.0 - torch.exp(- self.length * sigmas).mean())
        return {'sparsity': loss}


class CauchySparsityLoss(nn.Module):
    def __init__(self, weight) -> None:
        super(CauchySparsityLoss, self).__init__()
        self.weight = weight

    def forward(self, render_rets: List[RenderPassResult], batch):
        ret = [r for r in render_rets if 'sparsity_sample' == r.name]
        if 0 == len(ret):
            return {'sparsity': 0}
        sigmas = ret[0].sigmas
        loss = self.weight * torch.log(1+2*sigmas**2).mean()
        return {'sparsity': loss}
