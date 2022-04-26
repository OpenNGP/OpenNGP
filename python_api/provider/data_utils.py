import torch
import torch.nn.functional as F
import numpy as np

from python_api.utils.data_helper import namedtuple_map


def prepare_data(batch, device):
    if isinstance(batch['rays'].origins, np.ndarray):
        def map_func(r): return torch.from_numpy(
            r).to(device, non_blocking=True)
    else:
        def map_func(r): return r.to(device, non_blocking=True)

    for k, v in batch.items():
        if isinstance(v, tuple):
            batch[k] = namedtuple_map(map_func, v)
        else:
            batch[k] = map_func(v)
    return batch


def srgb_to_linear(srgb):
    linear = torch.zeros_like(srgb)
    cvt_mask = srgb <= 0.04045
    linear[cvt_mask] = srgb[cvt_mask] / 12.92
    linear[~cvt_mask] = ((srgb[~cvt_mask] + 0.055) / 1.055).pow(2.4)
    return linear


def linear_to_srgb(linear):
    srgb = torch.zeros_like(linear)
    cvt_mask = linear < 0.0031308
    srgb[cvt_mask] = linear[cvt_mask] * 12.92
    srgb[~cvt_mask] = 1.055 * linear[~cvt_mask].pow(0.41666) - 0.055
    return srgb



def depth_to_pts(img, depth, H, W, intrinsic, pose, is_ray_depth):
    pts = depth_to_pts_wo_color(depth, H, W, intrinsic, pose, is_ray_depth)
    return np.concatenate([pts, img], axis=2)


def depth_to_pts_wo_color(depth, H, W, intrinsic, pose, is_ray_depth):
    import torch
    rays_o, rays_d, _ = get_rays(torch.tensor(pose[None, ...]), torch.tensor(intrinsic[None, ...]), H, W, -1)
    rays_o = rays_o.reshape(1, H, W, -1).numpy().squeeze()
    rays_d = rays_d.reshape(1, H, W, -1).numpy().squeeze()
    if is_ray_depth:
        ray_depth = depth[..., None]
    else:
        ray_depth = depth[..., None] / np.matmul(rays_d, pose[:3, 2:3])
    # pts = rays_o + ray_depth * rays_d
    # pts = np.reshape(pts, (-1, 3))
    # colors = np.reshape(255*img, (-1, 3)).astype(np.uint8)
    return np.concatenate([rays_o + ray_depth * rays_d], axis=2)


def pts_to_mesh(pts, outfile):
    from trimesh import Trimesh
    from trimesh.exchange.ply import export_ply
    pts = pts.reshape((-1, pts.shape[-1]))
    mesh = Trimesh(vertices=pts[:, :3], vertex_colors=pts[:, 3:], process=False)
    open(outfile, 'wb').write(export_ply(mesh))


def lift(x, y, z, intrinsics):
    # x, y, z: [B, N]
    # intrinsics: [B, 3, 3]

    device = x.device
    
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    sk = intrinsics[..., 0, 1].unsqueeze(-1)

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)


def get_rays(c2w, intrinsics, H, W, N_rays=-1):
    # c2w: [B, 4, 4]
    # intrinsics: [B, 3, 3]
    # return: rays_o, rays_d: [B, N_rays, 3]
    # return: select_inds: [B, N_rays]

    device = c2w.device
    rays_o = c2w[..., :3, 3] # [B, 3]
    prefix = c2w.shape[:-2]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij') # for torch < 1.10, should remove indexing='ij'
    i = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        select_hs = torch.randint(0, H, size=[N_rays], device=device)
        select_ws = torch.randint(0, W, size=[N_rays], device=device)
        select_inds = select_hs * W + select_ws
        select_inds = select_inds.expand([*prefix, N_rays])
        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
    else:
        select_inds = torch.arange(H*W, device=device).expand([*prefix, H*W])

    pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsics)
    pixel_points_cam = pixel_points_cam.transpose(-1, -2)

    world_coords = torch.bmm(c2w, pixel_points_cam).transpose(-1, -2)[..., :3]
    
    rays_d = world_coords - rays_o[..., None, :]
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = rays_o[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds


def est_global_scale(frame_pts):
    all_pts = np.vstack(frame_pts)[:, :3]
    bmin = all_pts.min(axis=0)
    bmax = all_pts.max(axis=0)
    diag = np.sqrt(((bmax-bmin)*(bmax-bmin)).sum())
    center = 0.5*(bmin+bmax)
    print(f'diag: {diag}', f'center: {center}')
    # default bound 2.0, scene diag 4
    scale = 4 / diag
    offset = -scale*center
    print(f'scale: {scale}', f'offset: {offset}')
    return scale, offset.tolist()
