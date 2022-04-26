import torch.nn as nn

from typing import List
from python_api.metrics import Loss_factory
from python_api.renderer.renderpass import RenderPassResult


class CompositeLoss(nn.Module):
    def __init__(self, loss_configs) -> None:
        super(CompositeLoss, self).__init__()
        self.losses = {config['name']: Loss_factory.build(**config)
                       for config in loss_configs}
    
    def forward(self, render_rets: List[RenderPassResult], batch):
        loss_dct = {}
        for k, v in self.losses.items():
            loss_dct.update(v(render_rets, batch))
        return loss_dct
