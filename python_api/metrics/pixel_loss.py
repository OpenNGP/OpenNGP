import torch
import torch.nn as nn
import torch_scatter

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
            loss[f'loss_color_{ret.name}'] = self.criterion(ret.pixels.colors, pixels_gt)
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
            loss[f'loss_depth_{ret.name}'] = self.criterion(pred_depth, gt_depth) / self.bound
        return loss


class PixelLossWithDepthAndSight(nn.Module):
    def __init__(self, color_diff_type, depth_diff_type, bound, epsilon, decay) -> None:
        super(PixelLossWithDepthAndSight, self).__init__()
        self.color_and_depth = PixelLossWithDepth(color_diff_type, depth_diff_type, bound)
        self.epsilon = epsilon
        self.decay = decay
        self.step = 0

    def get_interval_inds(self, z_vals, target_z_vals, sample_per_ray):
        inds = torch.searchsorted(z_vals, target_z_vals)  # [N_ray, 1]
        inds = inds.squeeze()
        base = torch.arange(inds.shape[0], device=inds.device)
        base = base*sample_per_ray
        begin_idxes = inds.cumsum(dim=0).roll(1)
        begin_idxes[0] = 0
        result = torch.arange(inds.sum(), device=inds.device)
        result -= begin_idxes.repeat_interleave(inds)
        result += base.repeat_interleave(inds)
        return result

    def forward(self, render_rets: List[RenderPassResult], batch):
        loss_dct = self.color_and_depth(render_rets, batch)

        decay_rate = 0.1
        decay_steps = self.decay * 1000
        new_epsilon = self.epsilon * (decay_rate ** (self.step / decay_steps))
        
        # 
        ray_mask = batch['ray_mask']
        ray_depth = batch['ray_depth'][ray_mask.squeeze()]
        z_vals = render_rets[-1].samples.z_vals[ray_mask.squeeze()]  # [N_valid_ray, N_sample]
        weights = render_rets[-1].weights[ray_mask.squeeze()]  # [N_valid_ray, N_sample]

        inds = self.get_interval_inds(z_vals, ray_depth-self.epsilon, weights.shape[1])
        empty_weights = torch.gather(weights.reshape((1, 1, -1)),
                                     2,
                                     inds.reshape(1, 1, -1))
        loss_dct['loss_empty'] = empty_weights.square().sum() / weights.shape[0]
        return loss_dct


class PixelLossWithSightAndNear(nn.Module):
    def __init__(self, color_diff_type, depth_diff_type, bound, epsilon, decay) -> None:
        super(PixelLossWithSightAndNear, self).__init__()
        self.color = PixelLoss(color_diff_type)
        self.epsilon = epsilon
        self.decay = decay
        self.step = 0

    def get_interval_inds(self, lower_inds, inds, sample_per_ray):
        # inds = torch.searchsorted(z_vals, target_z_vals)  # [N_ray, 1]
        inds = inds.squeeze()
        base = torch.arange(inds.shape[0], device=inds.device)
        base = base*sample_per_ray+lower_inds.squeeze()
        begin_idxes = inds.cumsum(dim=0).roll(1)
        begin_idxes[0] = 0
        result = torch.arange(inds.sum(), device=inds.device)
        result -= begin_idxes.repeat_interleave(inds)
        result += base.repeat_interleave(inds)
        return result

    def forward(self, render_rets: List[RenderPassResult], batch):
        loss_dct = self.color(render_rets, batch)

        decay_rate = 0.1
        decay_steps = self.decay * 1000
        new_epsilon = self.epsilon * (decay_rate ** (self.step / decay_steps))
        
        # 
        ray_mask = batch['ray_mask']
        ray_depth = batch['ray_depth'][ray_mask.squeeze()]
        z_vals = render_rets[-1].samples.z_vals[ray_mask.squeeze()]  # [N_valid_ray, N_sample]
        weights = render_rets[-1].weights[ray_mask.squeeze()]  # [N_valid_ray, N_sample]

        inds = torch.searchsorted(z_vals, ray_depth-self.epsilon)
        gather_inds = self.get_interval_inds(torch.zeros_like(inds), inds, weights.shape[1])
        empty_weights = torch.gather(weights.reshape((1, 1, -1)),
                                     2,
                                     gather_inds.reshape(1, 1, -1))
        loss_dct['loss_empty'] = empty_weights.square().sum() / weights.shape[0]

        lower_inds = inds
        inds = torch.searchsorted(z_vals, ray_depth+self.epsilon)
        inds = inds - lower_inds
        gather_inds = self.get_interval_inds(lower_inds, inds, weights.shape[1])
        near_weights = torch.gather(weights.reshape((1, 1, -1)),
                                    2,
                                    gather_inds.reshape(1, 1, -1))

        add_inds = torch.arange(inds.shape[0], device=inds.device)
        add_inds = add_inds.repeat_interleave(inds.squeeze())
        loss_near = torch_scatter.scatter_add(near_weights, add_inds)
        loss_near = (1-loss_near).square().sum() / weights.shape[0]
        loss_dct['loss_near'] = loss_near

        return loss_dct
