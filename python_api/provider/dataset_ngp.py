from collections import namedtuple
import os
import time
import glob
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json

from tqdm import tqdm
from python_api.provider.data_utils import get_rays, cal_depth_confidences
from python_api.provider.scannet_utils import read_files

from python_api.renderer.rays import RaysWithDepth, RaysWithDepthCos


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0], coordinate='nerf'):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    if 'nerf' == coordinate:
        new_pose = np.array([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
            [0, 0, 0, 1],
        ])
    elif 'ngp' == coordinate:
        new_pose = np.array([
            [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[0]],
            [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
    elif 'nerf_synthetic' == coordinate:
        new_pose = np.array([
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
    else:
        raise NotImplementedError
    return new_pose  # [right, down, forward, pos]


def load_depth(datadir, fname):
    if os.path.exists(os.path.join(datadir, f'depth/{fname}.npy')):
        depth = np.load(os.path.join(datadir, f'depth/{fname}.npy'))
        # depth *= 10. # meter to decimeter
    else:
        with open(os.path.join(datadir, 'depth', f'{fname}.raw'), 'rb') as reader:
            depth = [np.array(reader.read(4)).view(dtype=np.float32)
                    for _ in range(256*192)]
            depth = np.array(depth).reshape((192, 256))
    return depth


def load_depth_confidence(datadir, fname):
    if os.path.exists(os.path.join(datadir, f'confidence/{fname}.npy')):
        mask = np.load(os.path.join(datadir, f'confidence/{fname}.npy'))
        confidence = np.zeros_like(mask, dtype=np.uint8)
        confidence[mask] = 2
    else:
        with open(os.path.join(datadir, 'confidence', f'{fname}.raw'), 'rb') as rd:
            confidence = [np.array(rd.read(1)).view(dtype=np.uint8)
                        for _ in range(256*192)]
            confidence = np.array(confidence).reshape((192, 256))
    return confidence


class NeRFDataset(Dataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = os.path.dirname(path)
        self.type = type
        self.downscale = downscale
        self.radius = radius # TODO: generate custom views for test?

        # load nerf-compatible format data.
        with open(path, 'r') as f:
            transform = json.load(f)

        self.near = transform.get('near', 0.0)
        self.far = transform.get('far', 0.0)
        coordinate = transform.get('coordinate', 'nerf')

        if 'nerf_synthetic' == coordinate:
            self.init_from_synthetic(transform, downscale, 'nerf', n_test)
        elif 'scannet' == coordinate:
            self.init_from_scannet(transform, downscale, 'nerf', n_test)
        elif 'muyu_synthetic' == coordinate:
            self.init_from_scannet(transform, downscale, 'ngp', n_test)
        else:
            self.init_from_arkit(transform, downscale, coordinate, n_test)

        vis_flag = False
        if vis_flag:
            self.vis_camera()

    def vis_camera(self):
        from .debug_utils import vis_camera, init_trescope
        from trescope import Trescope, Layout
        from trescope.toolbox import simpleDisplayOutputs
        from trescope.config import OrthographicCamera
        init_trescope(True, simpleDisplayOutputs(1, 1))
        Trescope().selectOutput(0).updateLayout(Layout().legendOrientation('vertical').camera(OrthographicCamera().eye(0, 0, 5).up(0, 1, 0)))
        for pi, pose in enumerate(self.poses):
            vis_camera(pose, 0, 0, 0, self.names[pi], 0)
        Trescope().breakPoint('')

    def init_from_arkit(self, transform, downscale, coordinate, n_test):
        type = self.type

        # load image size
        self.H = int(transform['h'] // downscale)
        self.W = int(transform['w'] // downscale)

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = transform['fl_x'] / downscale
        self.intrinsic[1, 1] = transform['fl_y'] / downscale
        self.intrinsic[0, 2] = transform['cx'] / downscale
        self.intrinsic[1, 2] = transform['cy'] / downscale

        self.scale = transform['scale']
        self.offset = transform['offset']
        self.depth_scale = transform['depth_scale']
        if transform.get('fusion') is not None:
            from trimesh import Trimesh
            from trimesh.exchange.obj import load_obj
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            mesh_path = os.path.join(self.root_path, transform.get('fusion'))
            mesh = load_obj(open(mesh_path))
            mesh = Trimesh(**mesh, process=False)
            self.fusion_intersector = RayMeshIntersector(mesh, scale_to_box=False)
        else:
            self.fusion_intersector = None

        if type == 'ar':
            self.poses = []
            self.image_names = []
            self.names = []
            frames = json.load(open(os.path.join(self.root_path, 'ar/render_frames.json')))['render_frames']
            for f in frames:
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                self.poses.append(pose)
                self.image_names.append(f['file_path'])
                name = os.path.splitext(os.path.basename(f['file_path']))
                self.names.append(name[0])
        elif type == 'debug':
            self.poses = []
            self.image_names = []
            self.names = []
            frames = json.load(open(os.path.join(self.root_path, 'debug_frames.json')))['render_frames']
            for f in frames:
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                self.poses.append(pose)
                self.image_names.append(f['file_path'])
                name = os.path.splitext(os.path.basename(f['file_path']))
                self.names.append(name[0])
        else:
            with open(os.path.join(self.root_path, 'train.txt')) as rd:
                lines = rd.readlines()
                names = [l[:-5] for l in lines]
                frame_train_idx = {n: ni for ni, n in enumerate(names)}

            # same order with poses_bounds.npy
            frames = transform["frames"]
            for frame in frames:
                name = os.path.splitext(os.path.basename(frame['file_path']))
                frame['file_idx'] = frame_train_idx[name[0]]
                frame['file_name'] = name[0]
            frames = sorted(frames, key=lambda d: d['file_idx'])

            if type == 'test':
                test_frames = os.path.join(self.root_path, 'test_frames.json')
                if not os.path.exists(test_frames):
                    # choose two random poses, and interpolate between.
                    f0, f1 = np.random.choice(frames, 2, replace=False)
                    pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), self.scale, self.offset, coordinate) # [4, 4]
                    pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), self.scale, self.offset, coordinate) # [4, 4]
                    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    slerp = Slerp([0, 1], rots)

                    frames = {'render_frames': []}
                    for i in range(n_test + 1):
                        ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = slerp(ratio).as_matrix()
                        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                        frames['render_frames'].append({
                            'file_path': f'{i}.jpg',
                            'transform_matrix': pose.tolist()
                        })
                    json.dump(frames, open(test_frames, 'w'), indent=2)
                else:
                    frames = json.load(open(test_frames))

                self.poses = []
                for f in frames['render_frames']:
                    pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                    self.poses.append(pose)

            else:
                if type == 'train':
                    mid = len(frames) // 2
                    frames = frames[:mid] + frames[mid+1:]
                elif type == 'valid':
                    mid = len(frames) // 2
                    frames = frames[mid:mid+1]

                per_img_depth_scale = transform.get('per_img_depth_scale', {})

                self.poses = []
                self.images = []
                self.depths = []
                self.masks = []
                self.names = []
                self.ori_depths = []
                for f in tqdm(frames):
                    f_path = os.path.join(self.root_path, f['file_path'])

                    # there are non-exist paths in fox...
                    if not os.path.exists(f_path):
                        continue

                    self.names.append(f['file_name'])
                    pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
                    pose = nerf_matrix_to_ngp(pose, self.scale, self.offset, coordinate)
                    if 'pose_refine' in f:
                        pose = np.matmul(np.array(f['pose_refine']), pose)

                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    image = image.astype(np.float32) / 255 # [H, W, 3/4]

                    self.poses.append(pose)
                    self.images.append(image)

                    depth_scale = self.depth_scale
                    if f['file_name'] in per_img_depth_scale:
                        depth_scale = per_img_depth_scale[f['file_name']]

                    img_id = os.path.splitext(os.path.basename(f['file_path']))[0]
                    depth = load_depth(self.root_path, img_id)
                    mask = load_depth_confidence(self.root_path, img_id)
                    self.ori_depths.append((depth*depth_scale, mask))
                    depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                    depth *= depth_scale
                    mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                    mask = mask >= 2
                    self.depths.append(depth)
                    self.masks.append(mask)

            self.poses = np.stack(self.poses, axis=0).astype(np.float32)

    def init_from_synthetic(self, transform, downscale, coordinate, n_test):
        if 'test' == self.type:
            transform_path = os.path.join(self.root_path, 'transforms_test.json')
            if os.path.exists(transform_path):
                transform = json.load(open(transform_path))
        elif 'valid' == self.type:
            transform_path = os.path.join(self.root_path, 'transforms_val.json')
            if os.path.exists(transform_path):
                transform = json.load(open(transform_path))
            else:
                transform['frames'] = transform['frames'][:1]
        elif self.type == 'train':
            transform['frames'] = transform['frames'][1:]

        # load image size
        self.H = int(transform['h'] // downscale)
        self.W = int(transform['w'] // downscale)

        # load intrinsics
        camera_angle_x = float(transform['camera_angle_x'])
        self.intrinsic = np.eye(3, dtype=np.float32)
        if 'fl_x' in transform:
            self.intrinsic[0, 0] = transform['fl_x'] / downscale
            self.intrinsic[1, 1] = transform['fl_y'] / downscale
            self.intrinsic[0, 2] = transform['cx'] / downscale
            self.intrinsic[1, 2] = transform['cy'] / downscale
        else:
            focal = .5 * transform['h'] / np.tan(.5 * camera_angle_x)
            self.intrinsic[0, 0] = focal / downscale
            self.intrinsic[1, 1] = focal / downscale
            self.intrinsic[0, 2] = transform['w'] * 0.5 / downscale
            self.intrinsic[1, 2] = transform['h'] * 0.5 / downscale

        self.scale = transform.get('scale', 1.0)
        self.offset = transform.get('offset', [0., 0., 0.])
        self.depth_scale = transform.get('depth_scale', 1.0)
        self.fusion_intersector = None

        self.poses = []
        self.images = []
        self.depths = []
        self.masks = []
        self.names = []
        self.ori_depths = []
        # self.rand_offsets = []
        for f in transform['frames']:
            f_path = os.path.join(self.root_path, f['file_path'])

            # there are non-exist paths in fox...
            if not os.path.exists(f_path):
                continue

            self.names.append(os.path.splitext(os.path.basename(f_path))[0])
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            # rand_offset = (np.random.rand(3)*2-1)*0.01
            # self.rand_offsets.append(rand_offset)
            # offset = self.offset+rand_offset
            pose = nerf_matrix_to_ngp(pose, self.scale, self.offset, coordinate)
            if 'pose_refine' in f:
                pose = np.matmul(np.array(f['pose_refine']), pose)

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255 # [H, W, 3/4]

            self.poses.append(pose)
            self.images.append(image)

            depth = np.ones_like(image[:, :, 0])
            mask = depth > 0
            self.ori_depths.append((depth*self.depth_scale, mask))
            self.depths.append(depth)
            self.masks.append(mask)

        self.poses = np.stack(self.poses, axis=0).astype(np.float32)
        # self.rand_offsets = np.stack(self.rand_offsets, axis=0).astype(np.float32)

    def init_from_scannet(self, transform, downscale, coordinate, n_test):
        self.scale = transform.get('scale', 1.0)
        self.offset = transform.get('offset', [0, 0, 0])
        self.depth_scale = transform.get('depth_scale', 1.0)
        depth_filter = transform.get('depth_filter')
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)

        poses = []
        intrinsics = []
        self.images = []
        self.depths = []
        self.masks = []
        self.names = []
        self.ori_depths = []
        depth_scaling_factor = float(transform['depth_scaling_factor'])
        for frame in tqdm(transform['frames']):
            name = os.path.basename(frame['file_path'])
            self.names.append(os.path.splitext(name)[0])
            pose = np.array(frame['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, self.scale, self.offset, coordinate)
            poses.append(pose)

            fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
            intrinsic = np.eye(3, dtype=np.float32)
            intrinsic[0, 0] = fx / downscale
            intrinsic[1, 1] = fy / downscale
            intrinsic[0, 2] = cx / downscale
            intrinsic[1, 2] = cy / downscale
            intrinsics.append(intrinsic)

            if '' == frame['file_path']:
                continue

            if 'rgb' in frame['file_path']:
                filename = frame['file_path']
                filename = filename.replace("rgb", "target_depth")
                filename = filename.replace(".jpg", ".png")
            else:
                filename = frame['depth_file_path']

            image, depth = read_files(self.root_path,
                                      frame['file_path'],
                                      filename)
            H, W = image.shape[:2]
            self.H = int(H // downscale)
            self.W = int(W // downscale)

            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self.images.append(image)

            mask = np.zeros_like(depth)
            mask[depth > 0.5] = 1 # 0 values are invalid depth
            depth = (depth / depth_scaling_factor).astype(np.float32)
            depth = depth*self.depth_scale
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            if depth_filter is not None:
                ksize = depth_filter['ksize']
                percentile = depth_filter['grad_cut_percentile']
                gX = cv2.Sobel(depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
                gY = cv2.Sobel(depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
                grad = np.sqrt(gX**2+gY**2)
                mask[grad > np.percentile(grad, percentile)] = 0
            self.depths.append(depth)
            self.masks.append(mask>0.5)
            ori_mask = np.zeros_like(depth, dtype=np.uint8)
            ori_mask[mask>0.5] = 3
            self.ori_depths.append((depth, ori_mask))
        self.poses = np.stack(poses, axis=0).astype(np.float32)
        self.intrinsic = np.stack(intrinsics, axis=0).astype(np.float32)
        pass

    def get_intrinsic(self, idx):
        if 2 < len(self.intrinsic.shape):
            return self.intrinsic[idx]
        return self.intrinsic

    def get_intrinsics(self, inds):
        return np.stack([self.get_intrinsic(idx) for idx in inds])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.get_intrinsic(index),
            'index': index,
        }

        if self.type == 'test' or self.type == 'ar' or self.type == 'debug':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            results['ori_H'] = str(int(self.H*self.downscale))
            results['ori_W'] = str(int(self.W*self.downscale))
            
            return results
        else:
            results['shape'] = (self.H, self.W)
            results['image'] = self.images[index]
            results['depth'] = self.depths[index]
            results['mask'] = self.masks[index]
            results['name'] = self.names[index]
            return results

    def export_pointcloud(self, output_dir=None):
        from trimesh import Trimesh
        from trimesh.exchange.ply import export_ply
        from .data_utils import depth_to_pts

        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        frame_pts = []
        for data in tqdm(self):
            intrinsic = self.get_intrinsic(data['index'])
            name = data['name']
            pts = depth_to_pts(data['image'], data['depth'], self.H, self.W, intrinsic, data['pose'], False)
            pts = pts[data['mask']]
            frame_pts.append(pts)
            if output_dir is not None:
                mesh = Trimesh(vertices=pts[:, :3], vertex_colors=pts[:, 3:], process=False)
                open(os.path.join(output_dir, f'{name}.ply'), 'wb').write(export_ply(mesh))

        return frame_pts

    def depth_pointcloud(self):
        from .data_utils import depth_to_pts_wo_color

        HH, WW = self.ori_depths[0][0].shape
        frame_pts = []
        for data in tqdm(self):
            intrinsic = self.get_intrinsic(data['index'])
            intrinsic = intrinsic / (self.H // HH)
            intrinsic[2, 2] = 1
            depth, mask = self.ori_depths[data['index']]
            pts = depth_to_pts_wo_color(depth, HH, WW, intrinsic, data['pose'], False)
            pts = pts[mask >= 2]
            frame_pts.append(pts)
        return frame_pts


class NeRFRayDataset(Dataset):
    def __init__(self, split, train_dir, config):
        super().__init__()
        type = split
        if '.json' in train_dir:
            path = train_dir
        else:
            path = os.path.join(train_dir, 'transforms_aligned.json')
        downscale = 1 if config.factor == 0 else config.factor
        radius = 1
        n_test = 10
        self.img_dataset = NeRFDataset(path, type, downscale, radius, n_test)
        if config.topk_depth_consistency > 0:
            topk = config.topk_depth_consistency
            self.depth_std = self.calc_depth_consistency(topk)
        else:
            self.depth_std = None
        # generate all rays
        self.rays_per_img = self.img_dataset.H * self.img_dataset.W
        self.total_rays = len(self.img_dataset) * self.rays_per_img
        # self.generate_ray_bundle()

    @property
    def size(self):
        return len(self.img_dataset)

    def generate_ray_bundle(self):
        from .data_utils import get_rays

        intersector = self.img_dataset.fusion_intersector
        bundle_rays, bundle_depths, bundle_masks, bundle_inds = [], [], [], []
        print(f'<===== processing ray bundle {"wo" if intersector is None else "w"} fusion mesh =====>')
        for data in tqdm(self.img_dataset):
            H, W = data['shape'][0], data['shape'][1]
            intrinsic = torch.tensor(data['intrinsic'][None, ...])
            pose = torch.tensor(data['pose'][None, ...])
            rays_o, rays_d, _ = get_rays(pose, intrinsic, H, W, -1)
            if intersector is None:
                rays_o = rays_o.reshape(1, H, W, -1).numpy().squeeze()
                rays_d = rays_d.reshape(1, H, W, -1).numpy().squeeze()
                ray_depth = data['depth'][..., None]
                ray_depth = ray_depth / np.matmul(rays_d, data['pose'][:3, 2:3])
                ray_rgb = np.concatenate([rays_o, rays_d, data['image']], axis=2)
                ray_mask = data['mask'][..., None]
            else:
                rays_o = rays_o.numpy().squeeze()
                rays_d = rays_d.numpy().squeeze()
                ray_depth = np.zeros((rays_o.shape[0], 1), dtype=rays_o.dtype)
                f_inds, ray_inds, locs = intersector.intersects_id(
                    rays_o, rays_d, False, 1, True)
                loc_ray_cos = np.einsum('ij,ij->i',
                                        intersector.mesh.face_normals[f_inds],
                                        rays_d[ray_inds])
                ray_mask = loc_ray_cos < 0
                ray_inds = ray_inds[ray_mask]
                f_inds = f_inds[ray_mask]
                locs = locs[ray_mask]
                ray_depth[ray_inds] = np.linalg.norm(rays_o[ray_inds]-locs, axis=1, keepdims=True)
                ray_mask = ray_depth > 0
                rays_o = rays_o.reshape((H, W, -1))
                rays_d = rays_d.reshape((H, W, -1))
                ray_rgb = np.concatenate([rays_o, rays_d, data['image']], axis=2)
                ray_depth = ray_depth.reshape((H, W, -1))
                ray_mask = ray_mask.reshape((H, W, -1))

                # update to data['depth']
                img_depth = ray_depth * np.matmul(rays_d, data['pose'][:3, 2:3])
                self.img_dataset.depths[data['index']] = img_depth.squeeze()
                self.img_dataset.masks[data['index']] = ray_mask.squeeze()

            bundle_rays.append(ray_rgb)
            bundle_depths.append(ray_depth)
            bundle_masks.append(ray_mask)
            bundle_inds.append(np.full_like(ray_mask, data['index'], dtype=int))

            pass
        self.bundles = {'ray_rgb': np.stack(bundle_rays).reshape((-1, 9)),
                        'ray_depth': np.stack(bundle_depths).reshape((-1, 1)),
                        'ray_mask': np.stack(bundle_masks).reshape((-1, 1)),
                        'ray_idx': np.stack(bundle_inds).reshape((-1, 1))}
        print('<===== processing ray bundle done =====>')

    @staticmethod
    def lift(x, y, z, intrinsics):
        # x, y, z: [B, N]
        # intrinsics: [B, 3, 3]
        return np.array([
            (x - intrinsics[0, 2] + intrinsics[1, 2] * intrinsics[0, 1] / intrinsics[1, 1] - intrinsics[0, 1] * y / intrinsics[1, 1]) / intrinsics[0, 0] * z,
            (y - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
            1
        ], dtype=np.float32)

    def __len__(self):
        return len(self.img_dataset)*self.rays_per_img
        return self.bundles['ray_rgb'].shape[0]

    def __getitem__(self, index):
        # compute ray when quering
        img_idx = int(index/self.rays_per_img)
        intrinsic = self.img_dataset.get_intrinsic(img_idx)
        img_ij = index - img_idx*self.rays_per_img
        i = int(img_ij/self.img_dataset.W)  # y
        j = img_ij - i*self.img_dataset.W  # x
        pt_cam = NeRFRayDataset.lift(j, i, 1, intrinsic)
        c2w = self.img_dataset.poses[img_idx]
        ray_o = c2w[:3, 3]
        pt_world = np.matmul(c2w, pt_cam)[:3]
        ray_d = pt_world - ray_o
        ray_view = ray_d / np.linalg.norm(ray_d)
        rgb = self.img_dataset.images[img_idx][i, j]
        depth = self.img_dataset.depths[img_idx][[i], [j]]
        depth_coef = np.dot(ray_view, c2w[:3, 2])
        ray_depth = depth / depth_coef
        mask = self.img_dataset.masks[img_idx][[i], [j]]

        item = {
            'pixels': rgb,
            'ray_depth': ray_depth,
            'ray_mask': mask,
            'rays': RaysWithDepth(index,
                                  np.array(img_idx),
                                  ray_o,
                                  ray_d,
                                  ray_view,
                                  depth_coef,
                                  np.array(0),
                                  np.array([self.img_dataset.near], dtype=np.float32),
                                  np.array([self.img_dataset.far], dtype=np.float32),
                                  depth_coef,
                                  ray_depth,
                                  mask)
        }

        if self.depth_std is not None:
            depth_error_std = self.depth_std[img_idx][[i], [j]]
            item['depth_var'] = depth_error_std**2
            item['ray_depth_var'] = (depth_error_std / depth_coef)**2

        return item

        item = {
            'index': index,
            'ray_idx': np.array([img_idx]),
            'ray_rgb': np.concatenate([ray_o, ray_d, rgb]),
            'ray_depth': ray_depth,
            'ray_mask': mask,
        }
        return item

        # cache use too much memory
        item = {
            'index': index,
            'ray_rgb': self.bundles['ray_rgb'][index],
            'ray_depth': self.bundles['ray_depth'][index],
            'ray_mask': self.bundles['ray_mask'][index],
            'ray_idx': self.bundles['ray_idx'][index],
        }
        if hasattr(self.img_dataset, 'rand_offsets'):
            item['pose_move'] = self.img_dataset.rand_offsets[item['ray_idx']]
        return item

    def depth_pointcloud(self):
        return self.img_dataset.depth_pointcloud()

    def calc_depth_consistency(self, topk):
        depths, masks = list(zip(*self.img_dataset.ori_depths))
        depths = np.stack(depths)
        masks = np.stack(masks) >= 2
        T = self.img_dataset.poses

        img_H, img_W = self.img_dataset.depths[0].shape
        factor = img_H / depths.shape[1]

        i_train = np.arange(len(T))
        K = np.identity(4, dtype=depths.dtype)
        K = K[None, ...].repeat(len(T), axis=0)
        K[:, :3, :3] = self.img_dataset.get_intrinsics(i_train)
        K[:, [0, 1, 0, 1], [0, 1, 2, 2]] /= factor

        stds = cal_depth_confidences(torch.from_numpy(depths),
                                     torch.from_numpy(T),
                                     torch.from_numpy(K),
                                     i_train,
                                     topk=topk,
                                     mask=torch.from_numpy(masks))
        stds = [cv2.resize(std, (img_W, img_H)) for std in stds]
        return np.stack(stds)

class NeRFDatasetTestIter:
    def __init__(self, split, train_dir, config):
        self.render_path = config.render_path
        type = split
        if '.json' in train_dir:
            path = train_dir
        elif self.render_path:
            path = os.path.join(train_dir, 'transforms_video.json')
        else:
            path = os.path.join(train_dir, 'transforms_test.json')
        downscale = 1 if config.factor == 0 else config.factor
        radius = 1
        n_test = 10
        self.img_dataset = NeRFDataset(path, type, downscale, radius, n_test)
        self.batch_size = config.batch_size
        self.it = 0
    
    @property
    def size(self):
        return len(self.img_dataset)

    def __iter__(self): return self
    def __next__(self):
        if self.render_path:
            idx = self.it
            self.it += 1
        else:
            idx = 0

        data = self.img_dataset[idx]
        
        H, W = int(data['H']), int(data['W'])
        intrinsic = torch.tensor(data['intrinsic'][None, ...])
        pose = torch.tensor(data['pose'][None, ...])
        origins, directions, _ = get_rays(pose, intrinsic, H, W, -1)
        origins = origins.reshape(H, W, -1).cpu().numpy().squeeze()
        directions = directions.reshape(H, W, -1).cpu().numpy().squeeze()

        depth_cos = np.matmul(directions, data['pose'][:3, 2:3])

        ones = np.ones_like(origins[..., :1])
        rays = RaysWithDepthCos(
            origins=origins,
            directions=directions/depth_cos,
            viewdirs=directions,
            radii=ones,
            lossmult=ones,
            near=ones*self.img_dataset.near,
            far=ones*self.img_dataset.far,
            depth_cos=depth_cos)

        extra_info = {
            'intrinsic': data['intrinsic'],
            'pose': data['pose'],
            'size': np.array([H, W])
        }

        if self.render_path:
            return {'rays': rays, **extra_info}
        else:
            return {
                'rays': rays,
                'pixels': self.img_dataset.images[0]
            }
