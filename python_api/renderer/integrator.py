import torch

from collections import namedtuple
from python_api.primitive.component.geometry import sigma
from python_api.utils import FunctionRegistry


Pixel = namedtuple('Pixel', ('colors', 'depths'))


def integrate_weight(sigmas, deltas):
    alphas = 1.-torch.exp(-sigmas*deltas)
    ones = torch.ones((alphas.shape[0], 1), device=alphas.device)
    weights = alphas * torch.cumprod(torch.cat([ones, 1.-alphas + 1e-10], -1), -1)[:, :-1]
    return weights, Pixel(None, None)


def propogate_neus_sdf(sigmas):
    "propogate sdf by weight interface. Followed by neus_important_sampler"
    return sigmas, Pixel(None, None)


def integrate_neus(primitive):
    pass


def volume_integrator(sigmas, rgbs, deltas, z_vals):
    weights, _ = integrate_weight(sigmas, deltas)
    colors = torch.sum(weights[...,None] * rgbs, -2)  # [N_rays, 3]
    depths = torch.sum(weights * z_vals, -1)
    return weights, Pixel(colors, depths)


def depth_integrator(sigmas, deltas, z_vals):
    weights, _ = integrate_weight(sigmas, deltas)
    depths = torch.sum(weights * z_vals, -1)
    return weights, Pixel(None, depths)


def pass_through_integrator(sigmas, deltas, z_vals):
    return None, Pixel(None, None)


rayintegrator = FunctionRegistry(
    volume_integrator=volume_integrator,
    depth_integrator=depth_integrator,
    weight_integrator=integrate_weight,
    pass_through_integrator=pass_through_integrator,
    propogate_neus_sdf=propogate_neus_sdf
)
