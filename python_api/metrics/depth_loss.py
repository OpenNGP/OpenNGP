import torch
import torch.nn as nn
import torch_scatter

from typing import List
from python_api.renderer.rays import RaysWithDepth, RaysWithDepthCos
from python_api.renderer.renderpass import RenderPassResult



def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)


def compute_depth_loss(depth_map, z_vals, weights, target_mean, target_std, target_valid_depth):
    pred_mean = depth_map[target_valid_depth]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    z_vals = z_vals[target_valid_depth.squeeze()]
    weights = weights[target_valid_depth.squeeze()]
    pred_var = ((z_vals - pred_mean.unsqueeze(-1)).pow(2) * weights).sum(-1) + 1e-5
    # target_mean = target_depth[..., 0][target_valid_depth]
    # target_std = target_depth[..., 1][target_valid_depth]
    target_mean = target_mean[target_valid_depth]
    target_std = target_std[target_valid_depth]
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    # f = nn.HuberLoss(delta=0.1)
    # return f(pred_mean, target_mean) / 1.5
    f = nn.GaussianNLLLoss(eps=0.001)
    loss_coef = float(pred_mean.shape[0]) / float(target_valid_depth.shape[0])
    return loss_coef * f(pred_mean, target_mean, pred_var)


"""
"--invalidate_large_std_threshold", type=float, default=1.,
    help='invalidate completed depth values with standard deviation greater than threshold in m, \
    thresholds <=0 deactivate invalidation'
"""


class DepthNLLLoss(nn.Module):
    """NLL depth loss introduced in 'Dense Depth Priors for Neural Radiance Fields from Sparse Input Views', CVPR 2022
    """
    def __init__(self, weight, large_std_threshold, use_depth_cos) -> None:
        super(DepthNLLLoss, self).__init__()
        self.weight = weight
        self.large_std_threshold = large_std_threshold
        self.use_depth_cos = use_depth_cos
    
    def forward(self, render_rets: List[RenderPassResult], batch):
        loss = {}
        # only compute for last pass
        render_ret = render_rets[-1]
        weights = render_ret.weights
        z_vals = render_ret.samples.z_vals
        depths = render_ret.pixels.depths[..., None]

        target_valid_depth = batch['ray_mask']
        target_depth = batch['ray_depth']

        if self.use_depth_cos:
            depth_cos = batch['rays'].depth_cos[..., None]
            depths = depths * depth_cos
            z_vals = z_vals * depth_cos
            target_depth = target_depth * depth_cos
        
        ray_depth_var = batch['ray_depth_var'] 
        if self.large_std_threshold > 0:
            var_valid_depth = ray_depth_var < self.large_std_threshold
            target_valid_depth = target_valid_depth & var_valid_depth

        loss_val = compute_depth_loss(depths,
                                      z_vals,
                                      weights,
                                      target_depth,
                                      ray_depth_var,
                                      target_valid_depth)

        loss['loss_depth_gaussian_nll'] = self.weight*loss_val
        return loss


class SightAndNearLoss(nn.Module):
    def __init__(self, epsilon, decay) -> None:
        super(SightAndNearLoss, self).__init__()
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
        loss_dct = {}

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
