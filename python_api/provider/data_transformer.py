import torch
import numpy as np

from typing import Union
from torch.utils.data import DataLoader
from os.path import join as pjoin
from python_api.app.config import Config
from python_api.provider.data_utils import prepare_data
from python_api.provider.data_utils import srgb_to_linear, linear_to_srgb
from python_api.provider import data_parameter
from .dataset import Dataset


class DataTransformer:
    def __init__(self, dataset: Union[Dataset, DataLoader], device, config: Config) -> None:
        """a handler to deal with possible data related learnable parameters
        exposure/intrinsics/extrinsics/etc

        Parameters
        ----------
        dataset : Union[Dataset, DataLoader]
            to query frame data (usually pixel and its corresponding ray)
        """
        self.dataset = dataset
        self.device = device
        self.color_mode = config.color_mode
        self.exposure_refine = None
        self.depth_scale_refine = None
        self.pose_refine = None

        size = dataset.dataset.size if isinstance(dataset, DataLoader) else dataset.size

        if config.refine_exposure:
            self.exposure_refine = data_parameter.ExposureRefine(size)
            self.exposure_refine.to(device)

        if config.refine_depth_scale:
            self.depth_scale_refine = data_parameter.DepthScaleRefine()
            self.depth_scale_refine.to(device)

        if config.refine_extrinsic:
            affine_repr = config.refine_extrinsic_affine_repr
            self.pose_refine = data_parameter.PoseRefine(size, affine_repr)
            self.pose_refine.to(device)

        if config.refine_extrinsic_perturb_std > 0:
            std = config.refine_extrinsic_perturb_std
            self.pose_pertrub = data_parameter.PosePurterb(size, affine_repr, std)
            vars = self.pose_pertrub.vars
            np.savetxt(pjoin(config.exp_dir, 'pose_perturb.txt'), vars.numpy())
            self.pose_pertrub.vars = vars.to(device)
            # self.pose_refine.compare(self.pose_pertrub)
        else:
            self.pose_pertrub = None

    def preprocess_data(self, batch):
        """_summary_

        Parameters
        ----------
        batch : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        batch = prepare_data(batch, self.device)
        
        if 'linear' == self.color_mode:
            batch['pixels'] = srgb_to_linear(batch['pixels'])

        if self.pose_pertrub is not None:
            ray_idx = batch['rays'].ray_idx.squeeze()
            rots = self.pose_pertrub.get_rotations(ray_idx)
            trans = self.pose_pertrub.get_translations(ray_idx)
            origins = batch['rays'].origins[..., None]
            directions = batch['rays'].directions[..., None]
            viewdirs = batch['rays'].viewdirs[..., None]
            new_origins = torch.bmm(rots, origins).squeeze() + trans
            new_directions = torch.bmm(rots, directions).squeeze()
            new_viewdirs = torch.bmm(rots, viewdirs).squeeze()
            batch['rays'] = batch['rays']._replace(origins=new_origins,
                                                   directions=new_directions,
                                                   viewdirs=new_viewdirs)
            
            pass

        if self.pose_refine is not None:
            ray_idx = batch['rays'].ray_idx.squeeze()
            rots = self.pose_refine.get_rotation_matrices(ray_idx)
            trans = self.pose_refine.get_translations(ray_idx)
            origins = batch['rays'].origins[..., None]
            directions = batch['rays'].directions[..., None]
            viewdirs = batch['rays'].viewdirs[..., None]
            new_origins = torch.bmm(rots, origins).squeeze() + trans
            new_directions = torch.bmm(rots, directions).squeeze()
            new_viewdirs = torch.bmm(rots, viewdirs).squeeze()
            batch['rays'] = batch['rays']._replace(origins=new_origins,
                                                   directions=new_directions,
                                                   viewdirs=new_viewdirs)
            pass

        return batch

    def postprocess_data(self, render_rets, batch):
        """_summary_

        Parameters
        ----------
        render_rets : _type_
            _description_
        batch : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if self.exposure_refine is not None:
            ray_idx = batch['rays'].ray_idx[None, ...]
            expo_scales = self.exposure_refine.get_exposure_scale(ray_idx)
            for ri, ret in enumerate(render_rets):
                if ret.pixels.colors is None:
                    continue
                colors = ret.pixels.colors * expo_scales.squeeze()
                pixels = ret.pixels._replace(colors=colors)
                render_rets[ri] = ret._replace(pixels=pixels)

        if self.depth_scale_refine is not None:
            batch['ray_depth'] = batch['ray_depth']*self.depth_scale_refine.get_depth_scale()

        return render_rets

    def postprocess_data_eval(self, pred_rgb, batch):
        if self.exposure_refine is not None:
            if 'linear' == self.color_mode:
                gt_rgb = srgb_to_linear(batch['pixels'])
                pred_rgb = srgb_to_linear(pred_rgb)
            else:
                gt_rgb = batch['pixels']
            coefs = (gt_rgb*pred_rgb).reshape((-1, 3))
            denom = (pred_rgb*pred_rgb).reshape((-1, 3))
            s = coefs.sum(dim=0)/denom.sum(dim=0)
            if 'linear' == self.color_mode:
                pred_rgb = linear_to_srgb(pred_rgb*s)
            else:
                pred_rgb = pred_rgb*s
        return pred_rgb

    def collect_parameters(self):
        params = []
        if self.exposure_refine is not None:
            params += [{
                'name': 'exposure',
                'params': list(self.exposure_refine.parameters())
            }]

        if self.depth_scale_refine is not None:
            params += [{
                'name': 'depth_scale',
                'params': list(self.depth_scale_refine.parameters())
            }]

        if self.pose_refine is not None:
            params += [{
                'name': 'pose_refine',
                'params': list(self.pose_refine.parameters())
            }]

        return params

    def load_state(self, state_dict):
        if self.exposure_refine is not None:
            self.exposure_refine.load_state_dict(state_dict['exposure'])
        if self.depth_scale_refine is not None:
            self.depth_scale_refine.load_state_dict(state_dict['depth_scale'])
        if self.pose_refine is not None:
            self.pose_refine.load_state_dict(state_dict['pose_refine'])

    def export_state(self):
        states = {}
        if self.exposure_refine is not None:
            states['exposure'] = self.exposure_refine.state_dict()
        if self.depth_scale_refine is not None:
            states['depth_scale'] = self.depth_scale_refine.state_dict()
        if self.pose_refine is not None:
            states['pose_refine'] = self.pose_refine.state_dict()
        return states
