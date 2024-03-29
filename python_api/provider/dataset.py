# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different datasets implementation plus a general port for all the datasets."""
import json
import queue
import threading
import cv2
import numpy as np
from os import listdir
from os.path import exists, join as pjoin, dirname, isdir
from PIL import Image
from python_api.renderer.rays import Rays, RaysWithDepthCos, RaysWithDepthCos2
from python_api.utils.data_helper import namedtuple_map


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.split = split
    self.data_dir = data_dir if isdir(data_dir) else dirname(data_dir) 
    self.near = config.near
    self.far = config.far
    self.lazy_ray = config.lazy_ray
    if split == 'train':
      self._train_init(config)
    elif split == 'test':
      self._test_init(config)
    elif 'test' in split:
      self._test_init(config)
    else:
      raise ValueError(
          'the split argument should be either \'train\' or \'test\', set'
          'to {} here.'.format(split))
    # self.batch_size = config.batch_size // jax.host_count()
    self.batch_size = config.batch_size
    self.batching = config.batching
    self.render_path = config.render_path
    self.precrop_iters = config.precrop_iters
    self.precrop_frac = config.precrop_frac
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.get()
    return x
    # if self.split == 'train':
    #   return utils.shard(x)  # Split data into shards for multiple devices along the first dimension.
    # else:
    #   return utils.to_device(x)  # Transfer data to devices (GPU/TPU).

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    return x
    # if self.split == 'train':
    #   return utils.shard(x)
    # else:
    #   return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_rays()
    self.it = 0

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = self.images.reshape([-1, 3])
      self.rays = namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                       self.rays)
    elif config.batching == 'single_image':
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')

  def _test_init(self, config):
    self._load_renderings(config)
    if not self.lazy_ray:
      self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    self.it += 1
    if self.batching == 'all_images':
      ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[ray_indices]
      batch_rays = namedtuple_map(lambda r: r[ray_indices], self.rays)
    elif self.batching == 'single_image':
      if self.it <= self.precrop_iters:
        dH = int(self.h//2 * self.precrop_frac)
        dW = int(self.w//2 * self.precrop_frac)
        coords = np.stack(np.meshgrid(
          np.linspace(self.h//2 - dH, self.h//2 + dH - 1, 2*dH),
          np.linspace(self.w//2 - dW, self.w//2 + dW - 1, 2*dW)
        ), -1).reshape((-1, 2)).astype(int)  # [2*dH*2*dW, 2], [i, j]
        coords = coords[:, 0] * self.w + coords[:, 1]
        ray_indices = np.random.randint(0, coords.shape[0],
                                        (self.batch_size,))
        ray_indices = coords[ray_indices]
      else:
        ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                        (self.batch_size,))

      image_index = np.random.randint(0, self.n_examples, ())
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')

    return {'pixels': batch_pixels, 'rays': batch_rays}

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.lazy_ray:
      rays = self._generate_rays_img(idx)
      data = {'rays': namedtuple_map(lambda r: r[0], rays)}
      if not self.render_path:
        data['pixels'] = self.images[idx]
      return data

    if self.render_path:
      return {'rays': namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      return {
          'pixels': self.images[idx],
          'rays': namedtuple_map(lambda r: r[idx], self.rays)
      }

  def _generate_rays_from_cams(self, camtoworlds):
    """Generating rays for all images. coordinate [right|up|backward]"""
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - self.w * 0.5 + 0.5) / self.focal,
         -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, ..., None, :] *
                   camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    depth_cos = -(viewdirs*camtoworlds[:, None, None, :3, 2]).sum(axis=-1)

    ray_idx = np.arange(camtoworlds.shape[0])
    ray_idx = np.repeat(ray_idx[..., None], self.h, axis=-1)
    ray_idx = np.repeat(ray_idx[..., None], self.w, axis=-1)
    ray_idx = ray_idx[..., None]
    ones = np.ones_like(origins[..., :1])
    return RaysWithDepthCos2(
        ray_idx=ray_idx,
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far,
        depth_cos=depth_cos[..., None])

  def _generate_rays_img(self, idx):
    """Generating rays for all images. coordinate [right|up|backward]"""
    return self._generate_rays_from_cams(self.camtoworlds[idx:idx+1])

  # TODO(bydeng): Swap this function with a more flexible camera model.
  def _generate_rays(self):
    """Generating rays for all images. coordinate [right|up|backward]"""
    self.rays = self._generate_rays_from_cams(self.camtoworlds)


class Multicam(Dataset):
  """Multicam Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the Multicam dataset.')
    with open(pjoin(self.data_dir, 'metadata.json'),
                         'r') as fp:
      self.meta = json.load(fp)[self.split]
    self.meta = {k: np.array(self.meta[k]) for k in self.meta}
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    images = []
    for fbase in self.meta['file_path']:
      fname = pjoin(self.data_dir, fbase)
      with open(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
      if config.white_bkgd:
        image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
      images.append(image[..., :3])
    self.images = images
    self.n_examples = len(self.images)

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_rays()

    def flatten(x):
      # Always flatten out the height x width dimensions
      x = [y.reshape([-1, y.shape[-1]]) for y in x]
      if config.batching == 'all_images':
        # If global batching, also concatenate all data into one list
        x = np.concatenate(x, axis=0)
      return x

    self.images = flatten(self.images)
    self.rays = namedtuple_map(flatten, self.rays)

  def _test_init(self, config):
    self._load_renderings(config)
    self._generate_rays()
    self.it = 0

  def _generate_rays(self):
    """Generating rays for all images."""
    pix2cam = self.meta['pix2cam']
    cam2world = self.meta['cam2world']
    width = self.meta['width']
    height = self.meta['height']

    def res2grid(w, h):
      return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
          np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
          indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
    origins = [
        np.broadcast_to(c2w[:3, -1], v.shape)
        for v, c2w in zip(directions, cam2world)
    ]
    viewdirs = [
        v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
    ]

    def broadcast_scalar_attribute(x):
      return [
          np.broadcast_to(x[i], origins[i][..., :1].shape)
          for i in range(self.n_examples)
      ]

    lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
    near = broadcast_scalar_attribute(self.meta['near'])
    far = broadcast_scalar_attribute(self.meta['far'])

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1)) for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

    self.rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=lossmult,
        near=near,
        far=far)


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    # if config.render_path:
    #   raise ValueError('render_path cannot be used for the blender dataset.')
    with open(
        pjoin(self.data_dir, 'transforms_{}.json'.format(self.split)),
        'r') as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(len(meta['frames'])):
      frame = meta['frames'][i]
      fname = pjoin(self.data_dir, frame['file_path'] + '.png')
      with open(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if config.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif config.factor > 0:
          raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                           'set.'.format(config.factor))
      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if config.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta['camera_angle_x'])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]


class Muyu(Dataset):
  """Muyu Dataset."""

  @staticmethod
  def ngp_to_mipnerf(pose):
    pose = np.array(pose)
    new_pose = np.array([
      [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
      [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
      [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
      [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose  # [right, up, backward, pos]

  def _load_rendering_path(self, config, frames):
    cams = []
    for i in range(len(frames)):
      frame = frames[i]
      cams.append(Muyu.ngp_to_mipnerf(frame['transform_matrix']))
    return cams

  def _load_rendering_data(self, config, frames):
    images = []
    cams = []
    for i in range(len(frames)):
      frame = frames[i]
      fname = pjoin(self.data_dir, frame['file_path'])
      with open(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if config.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif config.factor > 0:
          raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                           'set.'.format(config.factor))
      cams.append(Muyu.ngp_to_mipnerf(frame['transform_matrix']))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if config.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    return cams, self.images.shape[1:3]

  def _load_renderings(self, config):
    """Load images from disk."""
    if '.json' in config.data_dir:
      fpath = config.data_dir
    else:
      fpath = pjoin(self.data_dir, 'transforms_aligned.json')
    with open(fpath, 'r') as fp:
      meta = json.load(fp)

    if 'train' == self.split or 'test_train' == self.split:
      mid = len(meta['frames']) // 2
      frames = meta['frames'][:mid] + meta['frames'][mid+1:]
      cams, hw = self._load_rendering_data(config, frames)
    elif 'test' == self.split:
      if config.render_path:
        frames_path = open(pjoin(self.data_dir, 'ar/render_frames.json'))
        frames = json.load(frames_path)['render_frames']
        cams = self._load_rendering_path(config, frames)
        hw = (int(meta['h']), int(meta['w']))
      else:
        mid = len(meta['frames']) // 2
        frames = meta['frames'][mid:mid+1]
        cams, hw = self._load_rendering_data(config, frames)
    else:
      raise ValueError('Unkown split for Muyu Dataset')

    self.h, self.w = hw
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta['camera_angle_x'])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.frames = frames
    self.n_examples = self.camtoworlds.shape[0]

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    super()._generate_rays()
    self.render_rays = self.rays

class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    # Load images.
    imgdir_suffix = ''
    if config.factor > 0:
      imgdir_suffix = '_{}'.format(config.factor)
      factor = config.factor
    else:
      factor = 1
    imgdir = pjoin(self.data_dir, 'images' + imgdir_suffix)
    if not exists(imgdir):
      raise ValueError('Image folder {} does not exist.'.format(imgdir))
    imgfiles = [
        pjoin(imgdir, f)
        for f in sorted(listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    images = []
    for imgfile in imgfiles:
      with open(imgfile, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        images.append(image)
    images = np.stack(images, axis=-1)

    # Load poses and bds.
    with open(pjoin(self.data_dir, 'poses_bounds.npy'),
                         'rb') as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[-1] != images.shape[-1]:
      raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
          images.shape[-1], poses.shape[-1]))

    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale

    # Recenter poses.
    poses = self._recenter_poses(poses)

    # Generate a spiral/spherical ray path for rendering videos.
    if config.spherify:
      poses = self._generate_spherical_poses(poses, bds)
      self.spherify = True
    else:
      self.spherify = False
    if not config.spherify and self.split == 'test':
      self._generate_spiral_poses(poses, bds)

    # Select the split.
    i_test = np.arange(images.shape[0])[::config.llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    if self.split == 'train':
      indices = i_train
    else:
      indices = i_test
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses[:, :3, :4]
    self.focal = poses[0, -1, -1]
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w
    if config.render_path:
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    if self.split == 'test':
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays()

    if not self.spherify:
      ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                   self.rays.directions,
                                                   self.focal, self.w, self.h)

      mat = ndc_origins
      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

      dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :])**2, -1))
      dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
      # Cut the distance in half, and then round it out so that it's
      # halfway between inscribed by / circumscribed about the pixel.
      radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

      ones = np.ones_like(ndc_origins[..., :1])
      self.rays = Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=self.rays.directions,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)

    # Split poses from the dataset and generated poses
    if self.split == 'test':
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split = [np.split(r, [n_render_poses], 0) for r in self.rays]
      split0, split1 = zip(*split)
      self.render_rays = Rays(*split0)
      self.rays = Rays(*split1)

  def _recenter_poses(self, poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = self._poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

  def _poses_avg(self, poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = self._normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
    return c2w

  def _viewmatrix(self, z, up, pos):
    """Construct lookat view matrix."""
    vec2 = self._normalize(z)
    vec1_avg = up
    vec0 = self._normalize(np.cross(vec1_avg, vec2))
    vec1 = self._normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

  def _normalize(self, x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

  def _generate_spiral_poses(self, poses, bds):
    """Generate a spiral path for rendering."""
    c2w = self._poses_avg(poses)
    # Get average pose.
    up = self._normalize(poses[:, :3, 1].sum(0))
    # Find a reasonable 'focus depth' for this dataset.
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    # Get radii for spiral path.
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2
    # Generate poses for spiral path.
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    zrate = .5
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
      c = np.dot(c2w[:3, :4], (np.array(
          [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
      z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
      render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
    self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

  def _generate_spherical_poses(self, poses, bds):
    """Generate a 360 degree spherical path for rendering."""
    # pylint: disable=g-long-lambda
    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
      a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
      b_i = -a_i @ rays_o
      pt_mindist = np.squeeze(-np.linalg.inv(
          (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
      return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = self._normalize(up)
    vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
    vec2 = self._normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
      camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
      up = np.array([0, 0, -1.])
      vec2 = self._normalize(camorigin)
      vec0 = self._normalize(np.cross(vec2, up))
      vec1 = self._normalize(np.cross(vec2, vec0))
      pos = camorigin
      p = np.stack([vec0, vec1, vec2, pos], 1)
      new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    if self.split == 'test':
      self.render_poses = new_poses[:, :3, :4]
    return poses_reset
