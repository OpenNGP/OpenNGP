import torch
import numpy as np
from collections import namedtuple
from python_api.renderer.raymarching import near_far_from_aabb
from python_api.renderer.rays import Rays, RaysWithDepth
from python_api.utils import FunctionRegistry


SamplerResult = namedtuple(
    'SamplerResult',
    ('xyzs', 'views', 'z_vals', 'deltas')
)

SamplerResultWithBound = namedtuple(
    'SamplerResultWithBound',
    ('xyzs', 'views', 'z_vals', 'deltas', 'nears', 'fars')
)


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    device = weights.device
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def delta_from_zval(z_vals, rays_d, delta_inf=1e10):
    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    delta_inf = delta_inf * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    deltas = deltas * torch.norm(rays_d[...,None,:], dim=-1)
    return deltas


def uniform_sampler(rays: Rays, N_samples: int, lindisp: bool, perturb: bool):
    near, far = rays.near, rays.far
    rays_o, rays_d = rays.origins, rays.directions
    N_rays = rays_o.shape[0]
    device = rays_o.device

    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    
    # Convert these values using volume rendering (Section 4)
    deltas = delta_from_zval(z_vals, rays_d)

    views = rays.viewdirs[...,None,:].expand(pts.shape)

    return SamplerResult(pts, views, z_vals, deltas)


def neus_sampler(rays: Rays, N_samples: int, N_samples_outside: int, perturb: bool,
                up_sample_steps:int):
    near, far = rays.near, rays.far  # [N_rays,], [N_rays,]
    rays_o, rays_d = rays.origins, rays.directions  # [N_rays, 3], [N_rays, 3]
    N_rays = rays_o.shape[0]
    device = rays_o.device
    sample_dist = 2.0 / N_samples   # Assuming the region of interest is a unit sphere
    z_vals = torch.linspace(0.0, 1.0, N_samples)
    z_vals = near + (far - near) * z_vals[None, :]
    if perturb > 0:
        t_rand = (torch.rand([N_rays, 1]) - 0.5)
        z_vals = z_vals + t_rand * 2.0 / N_samples_outside

    ### outside sampling
    z_vals_outside = None
    if N_samples_outside > 0:
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (N_samples_outside + 1.0), N_samples_outside)  # confusing
        if perturb > 0:
            mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([N_rays, z_vals_outside.shape[-1]])
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / N_samples  # range from far to infinity

    ### inside sampling
    if N_samples > 0:
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        # TODO query sdf
        sdf = pts.sum(dim=-1)  # [N_rays, N_samples]
        for idx in range(up_sample_steps):
            new_z_vals = neus_upsample(rays_o, rays_d, z_vals, sdf, 
                                        n_importance=N_samples//up_sample_steps, inv_s=64 * 2**idx)
    pass


def neus_upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
    """
    Up sampling give a fixed inv_s
    """
    N_rays, N_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # N_rays, N_samples, 3
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
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
    prev_cos_val = torch.cat([torch.zeros([N_rays, 1]), cos_val[:, :-1]], dim=-1)
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
        torch.cat([torch.ones([N_rays, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples


def cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        # TODO: query sdf of new points
        # new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf


def _importance_sampler(rays: Rays,
                    samples: SamplerResult,  # samples from last pass
                    weights,
                    N_importance,
                    use_norm_dir,
                        delta_inf,
                        perturb,
                        concat_input_sample,
                        stop_grad):
    rays_o = rays.origins
    if use_norm_dir:
        rays_d = rays.viewdirs
    else:
        rays_d = rays.directions
    z_vals = samples.z_vals
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))

    if stop_grad:
        z_samples = z_samples.detach()

    if concat_input_sample:
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    else:
        z_vals, _ = torch.sort(z_samples, -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

    # Convert these values using volume rendering (Section 4)
    deltas = delta_from_zval(z_vals, rays_d, delta_inf)

    views = rays.viewdirs[...,None,:].expand(pts.shape)
    
    return SamplerResult(pts, views, z_vals, deltas)
                   

def importance_sampler(rays: Rays,
                       samples: SamplerResult,  # samples from last pass
                       weights,
                       N_importance,
                       use_norm_dir,
                       perturb):
    return _importance_sampler(rays,
                               samples,
                               weights,
                               N_importance,
                               use_norm_dir,
                               1e10,
                               perturb,
                               True,
                               True)


def importance_sampler_mipnerf360(rays: Rays,
                                  samples: SamplerResult,  # samples from last pass
                                  weights,
                                  N_importance,
                                  use_norm_dir,
                                  perturb,
                                  concat_input_sample,
                                  stop_grad):
    return _importance_sampler(rays,
                               samples,
                               weights,
                               N_importance,
                               use_norm_dir,
                               1e10,
                               perturb,
                               concat_input_sample,
                               stop_grad)


def ngp_importance_sampler(rays: Rays,
                           samples: SamplerResultWithBound,  # samples from last pass
                           weights,
                           primitive,
                           N_importance,
                           perturb):
    if isinstance(samples, SamplerResultWithBound):
        sample_dist = (samples.fars - samples.nears) / samples.z_vals.shape[-1]
        use_norm_dir = True
    else:
        sample_dist = (rays.far - rays.near) / samples.z_vals.shape[-1]
        use_norm_dir = False

    sample_ret = _importance_sampler(rays,
                               samples,
                               weights,
                               N_importance,
                               use_norm_dir,
                               sample_dist,
                               perturb,
                               True,
                               True)
    aabb = primitive.geometry.aabb
    pts = torch.min(torch.max(sample_ret.xyzs, aabb[:3]), aabb[3:]) # a manual clip.
    return sample_ret._replace(xyzs=pts)


def ngp_uniform_sampler(rays: Rays, primitive, num_steps, min_near, perturb):
    rays_o, rays_d = rays.origins, rays.viewdirs  # rays.directions isn't normalized
    device = rays_o.device
    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)

    N = rays_o.shape[0] # N = B * N, in fact
    device = rays_o.device

    # choose aabb
    aabb = primitive.geometry.aabb

    # sample steps
    nears, fars = near_far_from_aabb(rays_o, rays_d, aabb, min_near)
    nears.unsqueeze_(-1)
    fars.unsqueeze_(-1)

    #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
    z_vals = z_vals.expand((N, num_steps)) # [N, T]
    z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

    # same perturb strategy with ori nerf, which guarantees z_vals in [near, far] 
    # if perturb > 0.:
    #     # get intervals between samples
    #     mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    #     upper = torch.cat([mids, z_vals[...,-1:]], -1)
    #     lower = torch.cat([z_vals[...,:1], mids], -1)
    #     # stratified samples in those intervals
    #     t_rand = torch.rand(z_vals.shape, device=device)

    #     z_vals = lower + (upper - lower) * t_rand

    # perturb z_vals
    sample_dist = (fars - nears) / (num_steps - 1)
    if perturb:
        z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
        #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

    # generate xyzs
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
    pts = torch.min(torch.max(pts, aabb[:3]), aabb[3:]) # a manual clip.

    # Convert these values using volume rendering (Section 4)
    deltas = delta_from_zval(z_vals, rays_d, sample_dist)

    views = rays.viewdirs[...,None,:].expand(pts.shape)

    return SamplerResultWithBound(pts, views, z_vals, deltas, nears, fars)


def ngp_sampler_with_depth(rays: RaysWithDepth, primitive, num_steps, min_near, perturb, epsilon):
    rays_o, rays_d = rays.origins, rays.viewdirs  # rays.directions isn't normalized
    device = rays_o.device
    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)

    N = rays_o.shape[0] # N = B * N, in fact
    device = rays_o.device

    # choose aabb
    aabb = primitive.geometry.aabb

    # sample steps
    nears, fars = near_far_from_aabb(rays_o, rays_d, aabb, min_near)
    nears.unsqueeze_(-1)
    fars.unsqueeze_(-1)

    #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
    z_vals = z_vals.expand((N, num_steps)) # [N, T]
    z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

    # perturb z_vals
    sample_dist = (fars - nears) / num_steps
    if perturb:
        z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
        #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

    # sample from depth prior, N(d_prior, epsilon)
    extra_z_vals = torch.zeros_like(z_vals)
    valid_depth = rays.depth[rays.mask]
    m = torch.distributions.Normal(valid_depth, epsilon*torch.ones_like(valid_depth))
    depth_samples = m.sample(torch.Size([num_steps]))
    extra_z_vals[rays.mask.squeeze()] = depth_samples.T

    # sample from
    m = torch.distributions.Uniform(nears[~rays.mask], fars[~rays.mask])
    pad_samples = m.sample(torch.Size([num_steps]))
    extra_z_vals[~rays.mask.squeeze()] = pad_samples.T

    z_vals, _ = torch.sort(torch.cat([z_vals, extra_z_vals], -1), -1)

    # generate xyzs
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
    pts = torch.min(torch.max(pts, aabb[:3]), aabb[3:]) # a manual clip.

    # Convert these values using volume rendering (Section 4)
    deltas = delta_from_zval(z_vals, rays_d, sample_dist)

    views = rays.viewdirs[...,None,:].expand(pts.shape)

    return SamplerResult(pts, views, z_vals, deltas)


def sparsity_sampler(rays: Rays, primitive, n_sp):
    device = rays.origins.device
    bound = primitive.geometry.bound
    sp_points = torch.empty((n_sp, 3), device=device)
    sp_points.uniform_(-bound, bound)
    return SamplerResult(sp_points, None, None, None)


raysampler = FunctionRegistry(
    uniform_sampler=uniform_sampler,
    importance_sampler=importance_sampler,
    importance_sampler_mipnerf360=importance_sampler_mipnerf360,
    instant_ngp_sampler=ngp_uniform_sampler,
    ngp_uniform_sampler=ngp_uniform_sampler,
    ngp_importance_sampler=ngp_importance_sampler,
    ngp_sampler_with_depth=ngp_sampler_with_depth,
    sparsity_sampler=sparsity_sampler,
    neus_sampler=neus_sampler
)
