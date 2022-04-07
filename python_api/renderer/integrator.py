import torch

from collections import namedtuple
from python_api.utils import FunctionRegistry


Pixel = namedtuple('Pixel', ('colors', 'depths'))


def volume_integrator(sigmas, rgbs, deltas, z_vals):
    alphas = 1.-torch.exp(-sigmas*deltas)
    weights = alphas * torch.cumprod(torch.cat([torch.ones((alphas.shape[0], 1)), 1.-alphas + 1e-10], -1), -1)[:, :-1]
    colors = torch.sum(weights[...,None] * rgbs, -2)  # [N_rays, 3]
    depths = torch.sum(weights * z_vals, -1)
    return weights, Pixel(colors, depths)


rayintegrator = FunctionRegistry(
    volume_integrator=volume_integrator
)
