import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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


def calculate_coords(W, H, device):
    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
    pix_coords = torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0)
    ones = torch.ones(1, H * W, device=device)
    pix_coords = pix_coords.to(ones.device)
    pix_coords = torch.cat([pix_coords, ones], 0)
    return pix_coords


def BackprojectDepth(depth, invK, pix_coords):
    ## use cpu memory
    # batch_size, H, W = depth.shape
    # ones = np.ones((batch_size, 1, H * W))
    # cam_points = np.matmul(invK[:, :3, :3].cpu().numpy(), pix_coords.cpu().numpy())
    # cam_points = depth.view(batch_size, 1, -1).cpu().numpy() * cam_points
    # cam_points = np.concatenate([cam_points, ones], axis=1)
    # return torch.from_numpy(cam_points).to(depth.device)

    batch_size, H, W = depth.shape
    ones = torch.ones(batch_size, 1, H * W, device=depth.device)
    pix_coords = torch.matmul(invK[:, :3, :3], pix_coords)
    pix_coords = depth.view(batch_size, 1, -1) * pix_coords
    pix_coords = torch.cat([pix_coords, ones], 1)
    return pix_coords


def Project3D(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords


def Project3D_depth(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)
    return cam_points[:, 2, :].view(batch_size, H, W)


def cal_depth_confidences(depths, T, K, i_train, topk=4, mask=None):
    device = depths.device
    
    _, H, W = depths.shape
    view_num = len(i_train)
    if len(T.shape) > 2:
        batch_K = K[i_train]
        batch_invK = torch.inverse(batch_K)
    else:
        invK = torch.inverse(K)
        batch_K = torch.unsqueeze(K, 0).repeat(view_num, 1, 1)
        batch_invK = torch.unsqueeze(invK, 0).repeat(depths.shape[0], 1, 1)
    T_train = T[i_train]
    invT = torch.inverse(T_train)
    pix_coords = calculate_coords(W, H, device)
    cam_points = BackprojectDepth(depths, batch_invK, pix_coords)
    init_invalid_mask = torch.full((view_num, H, W), False, dtype=torch.bool, device=device)
    if mask is not None:
        init_invalid_mask = ~mask[i_train]

    depth_confidences = []
    for i in tqdm(range(depths.shape[0])):
        # depth_confidence = torch.zeros((H, W), dtype=torch.float32)
        # depth_confidences.append(depth_confidence)
        # continue

        cam_points_i = cam_points[i:i+1].repeat(view_num, 1, 1)
        T_i = torch.matmul(invT, T[i:i+1].repeat(view_num, 1, 1))
        pix_coords_ref = Project3D(cam_points_i, batch_K, T_i, H, W)
        depths_ = Project3D_depth(cam_points_i, batch_K, T_i, H, W)
        # zero padding will affect value of boarder pixel
        # 可以通过不取当前帧规避zero插值问题以及topk第一个始终为自己
        depths_proj = F.grid_sample(depths[i_train].unsqueeze(1),
                                    pix_coords_ref,
                                    padding_mode="border").squeeze()
        # make depths_proj out of border NaN
        invalid_mask = init_invalid_mask
        invalid_mask |= init_invalid_mask[i:i+1].expand(init_invalid_mask.shape)
        invalid_mask |= (pix_coords_ref[..., 0] < -1)
        invalid_mask |= (pix_coords_ref[..., 0] > 1)
        invalid_mask |= (pix_coords_ref[..., 1] < -1)
        invalid_mask |= (pix_coords_ref[..., 1] > 1)
        # depths_ might be negative!!!
        # error = torch.abs((depths_proj - depths_) / (depths_ + 1e-7))
        # 误差来源
        # 1. 本图深度突变采样误差
        # 2. 多图的不一致误差
        error = torch.abs(depths_proj - depths_)
        error[invalid_mask] = 1e10
        depth_confidence, top_inds = error.topk(k=topk, dim=0, largest=False)
        top_invalid_mask = torch.gather(invalid_mask, 0, top_inds)
        depth_confidence[top_invalid_mask] = torch.nan
        depth_confidence = torch.nanmean(depth_confidence, dim=0)
        depth_confidence[torch.isnan(depth_confidence)] = 0
        # depth_confidence = depth_confidence.mean(0).cpu().numpy()
        # debug, 0 uncertainty should resemble |d_pred-d_target|
        depth_confidences.append(depth_confidence)
    return np.stack(depth_confidences, 0)
    return np.stack(depth_confidences, 0)
