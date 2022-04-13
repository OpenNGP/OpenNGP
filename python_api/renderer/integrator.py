import torch

from collections import namedtuple
from python_api.utils import FunctionRegistry


Pixel = namedtuple('Pixel', ('colors', 'depths'))


def integrate_weight(sigmas, deltas):
    alphas = 1.-torch.exp(-sigmas*deltas)
    ones = torch.ones((alphas.shape[0], 1), device=alphas.device)
    weights = alphas * torch.cumprod(torch.cat([ones, 1.-alphas + 1e-10], -1), -1)[:, :-1]
    return weights, Pixel(None, None)


def volume_integrator(sigmas, rgbs, deltas, z_vals):
    weights, _ = integrate_weight(sigmas, deltas)
    colors = torch.sum(weights[...,None] * rgbs, -2)  # [N_rays, 3]
    depths = torch.sum(weights * z_vals, -1)
    return weights, Pixel(colors, depths)


def depth_integrator(sigmas, deltas, z_vals):
    weights, _ = integrate_weight(sigmas, deltas)
    depths = torch.sum(weights * z_vals, -1)
    return weights, Pixel(None, depths)


rayintegrator = FunctionRegistry(
    volume_integrator=volume_integrator,
    depth_integrator=depth_integrator,
    weight_integrator=integrate_weight
)
