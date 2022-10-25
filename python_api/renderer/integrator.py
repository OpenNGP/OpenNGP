import torch
import torch.nn.functional as F

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


def integrate_neus_upsample_weight(xyzs, sigmas, z_vals, idx):
    sdf = sigmas
    inv_s = 64 * 2 ** idx
    N_rays, N_samples = xyzs.shape[:2]
    radius = torch.linalg.norm(xyzs, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
    sdf = sdf.reshape(N_rays, N_samples)  # N_rays, N_samples
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([N_rays, 1]).to(cos_val.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([N_rays, 1]).to(alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    return weights, Pixel(None, None)


def integrate_neus(primitive, xyzs, deltas, views, sigmas, rgbs, geo_features):
    sdf = sigmas
    normals = geo_features[...,:3]
    ### calculate weight
    inv_s = primitive.query_shaped_invs(sdf.shape).reshape(-1,1)

    true_cos = (views * normals).sum(-1)

    cos_anneal_ratio = 1.0  # TODO add decay ratio with iter_step
    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    dists = deltas
    # Estimate signed distances at section points
    estimated_next_sdf = (sdf + iter_cos * dists).reshape(-1, 1) * 0.5
    estimated_prev_sdf = (sdf - iter_cos * dists).reshape(-1, 1) * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(*sdf.shape).clip(0.0, 1.0)

    pts_norm = torch.linalg.norm(xyzs, ord=2, dim=-1, keepdim=True).reshape(*sdf.shape)
    inside_sphere = (pts_norm < 1.0).float().detach()
    relax_inside_sphere = (pts_norm < 1.2).float().detach()

    batch_size = sdf.shape[0]
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    ### weighted sum
    colors = (rgbs * weights[:, :, None]).sum(dim=1)
    return weights, Pixel(colors, None)


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
    propogate_neus_sdf=propogate_neus_sdf,
    integrate_neus_upsample_weight=integrate_neus_upsample_weight,
    integrate_neus=integrate_neus
)
