from turtle import forward
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


class EikonalLoss(nn.Module):
    def __init__(self, igr_weight) -> None:
        super(EikonalLoss, self).__init__()
        self.igr_weight = igr_weight

    def forward(self, render_rets: List[RenderPassResult], batch):
        normals = render_rets[0].geo_features[...,:3]
        pts = render_rets[0].samples.xyzs # [bs, n_samples, 3]
        loss = (torch.linalg.norm(normals.reshape(*pts.shape), ord=2, dim=-1) - 1.0) ** 2
        batch_size, n_samples = pts.shape[:2]
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        loss = (relax_inside_sphere * loss).sum() / (relax_inside_sphere.sum() + 1e-5)
        return {"loss_eikonal": loss}