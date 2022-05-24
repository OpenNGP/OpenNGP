import torch
import torch.nn as nn

from typing import List
from python_api.renderer.renderpass import RenderPassResult


class IntervalDistLoss(nn.Module):
    """Regularization to overcome 'floater' and 'background collapse'
    [1] Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields, CVPR 2022 oral
    """
    def __init__(self, weight) -> None:
        super(IntervalDistLoss, self).__init__()
        self.weight = weight
    
    def forward(self, render_rets: List[RenderPassResult], batch):
        loss = {}
        # only compute for last pass
        render_ret = render_rets[-1]
        samples, weights = render_ret.samples, render_ret.weights
        mids = samples.z_vals + 0.5*samples.deltas
        coef = weights.unsqueeze(2)*weights.unsqueeze(1)
        term_0 = (coef*torch.abs(mids.unsqueeze(2)-mids.unsqueeze(1))).sum()
        term_1 = (torch.diagonal(coef, dim1=1, dim2=2)*samples.deltas).sum()

        loss['loss_interval_dist'] = self.weight*(term_0/2 + term_1/3) / batch['pixels'].shape[0]
        return loss
