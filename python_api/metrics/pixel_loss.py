from turtle import forward
import torch
import torch.nn as nn

from typing import List
from python_api.renderer.renderpass import RenderPassResult


class PixelLoss(nn.Module):
    def __init__(self, diff_type) -> None:
        super(PixelLoss, self).__init__()
        self.diff_type = diff_type
        if 'huber' == diff_type:
            self.criterion = torch.nn.HuberLoss(delta=0.1)
        elif 'l2' == diff_type:
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError
    
    def forward(self, render_rets: List[RenderPassResult], batch):
        pixels_gt = batch['pixels']
        loss = {}
        for ret in render_rets:
            if ret.pixels.colors is None:
                continue
            loss[f'{ret.name}/color'] = self.criterion(ret.pixels.colors, pixels_gt)
        return loss

class PixelLossWithDepth(nn.Module):
    def __init__(self, color_diff_type, depth_diff_type, bound) -> None:
        super(PixelLossWithDepth, self).__init__()
        self.color = PixelLoss(color_diff_type)
        self.depth_diff_type = depth_diff_type
        self.bound = bound
        if 'huber' == depth_diff_type:
            self.criterion = torch.nn.HuberLoss(delta=0.1)
        elif 'l2' == depth_diff_type:
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError

    def forward(self, render_rets: List[RenderPassResult], batch):
        depth_gt = batch['ray_depth']
        depth_mask = batch['ray_mask']
        loss = self.color(render_rets, batch)
        for ret in render_rets:
            if ret.pixels.depths is None:
                continue
            pred_depth = ret.pixels.depths[..., None][depth_mask]
            gt_depth = depth_gt[depth_mask]
            loss[f'{ret.name}/depth'] = self.criterion(pred_depth, gt_depth) / self.bound
