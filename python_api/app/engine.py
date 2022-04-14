import torch
import numpy as np
from rich.console import Console
from collections import namedtuple

from python_api.primitive import NGP_factory
from python_api.primitive.primitive import Primitive
from python_api.renderer.renderer import Renderer
from python_api.utils.data_helper import namedtuple_map


Pipeline = namedtuple('Pipeline', ('name', 'ngp', 'renderer'))


class Engine:
    def __init__(self, config) -> None:
        self.config = config
        self.console = Console()
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.primitives = {p['name']: self.build_primitive(p) for p in config.primitives}
        self.pipelines = [self.build_pipeline(p) for p in config.pipelines]
        self.eval_pipelines = [self.build_pipeline(p) for p in config.eval_pipelines]

    def build_primitive(self, primitive_def):
        name = primitive_def['name']
        ngp: Primitive = NGP_factory.build(**primitive_def['arch'])
        ngp.to(self.device)
        self.console.print(f'\[{name}] Geometry: ', ngp.geometry)
        self.console.print(f'\[{name}] Appearance: ', ngp.appearance)
        return ngp

    def build_pipeline(self, pipeline_def):
        name = pipeline_def['name']
        primitive = pipeline_def['primitive']
        renderer = Renderer(pipeline_def['renderpasses'])
        self.console.print(f'\[{name}] Renderer: ', renderer)
        return Pipeline(name, primitive, renderer)

    def collect_parameters(self):
        params = []
        for _, primitive in self.primitives.items():
            params += primitive.parameters()
        return params

    def load_ngp(self, state_dict):
        [p.load(state_dict[n]) for n, p in self.primitives.items()]

    def export_ngp(self):
        return {n: p.export() for n, p in self.primitives.items()}

    def execute_pipeline(self, pipelines, rays, context={}):
        rets, ctx = [], context
        for pipeline in pipelines:
            ngp = self.primitives[pipeline.ngp]
            rets += pipeline.renderer.render(rays, ngp, ctx)
            ctx.update(rets[-1]._asdict())
        return rets

    def run(self, rays, context={}):
        return self.execute_pipeline(self.pipelines, rays, context)

    def run_eval(self, rays, context={}):
        if 0 == len(self.eval_pipelines):
            return self.run(rays, context)
        else:
            return self.execute_pipeline(self.eval_pipelines, rays, context)

    def prepare_data(self, batch):
        device = self.device
        if isinstance(batch['rays'].origins, np.ndarray):
            map_func = lambda r: torch.from_numpy(r).to(device, non_blocking=True)
        else:
            map_func = lambda r: r.to(device, non_blocking=True)

        for k, v in batch.items():
            if isinstance(v, tuple):
                batch[k] = namedtuple_map(map_func, v)
            else:
                batch[k] = map_func(v)
        return batch

    def parse_loss_info(self, loss_dict):
        loss = None
        loss_stat = {}
        if isinstance(loss_dict, dict):
            for k, v in loss_dict.items():
                loss_stat[k] = v.item()
                if loss is None:
                    loss = v
                else:
                    loss += v
        else:
            loss = loss_dict
        loss_stat['loss'] = loss.item()
        return loss, loss_stat

    def draw(self, rays, chunk_size):
        # chunk = val_dataset.batch_size
        height, width = rays[0].shape[:2]
        num_rays = height * width
        test_rays = namedtuple_map(
            lambda r: r.reshape((num_rays, -1)),
            rays
        )
        rgbs, depths = [], []
        with torch.no_grad():
            for i in range(0, num_rays, chunk_size):
                # pylint: disable=cell-var-from-loop
                chunk_rays = namedtuple_map(
                    lambda r: r[i:i + chunk_size],
                    test_rays
                )
                rets_test = self.run_eval(chunk_rays, {'perturb': False})
                rgbs.append(rets_test[-1].pixels.colors)
                depths.append(rets_test[-1].pixels.depths)
            img_rgb = torch.concat(rgbs)
            img_rgb = img_rgb.reshape((height, width, -1))
            img_depth = torch.concat(depths)
            img_depth = img_depth.reshape((height, width, -1))
        return img_rgb, img_depth

    @staticmethod
    def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
        """Visualize the depth map with colormap.
        Rescales the values so that depth_min and depth_max map to 0 and 1,
        respectively.
        """
        import cv2
        if not direct:
            depth = 1.0 / (depth + 1e-6)
        invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
        if mask is not None:
            invalid_mask += np.logical_not(mask)
        if depth_min is None:
            depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
        if depth_max is None:
            depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
        depth[depth < depth_min] = depth_min
        depth[depth > depth_max] = depth_max
        depth[invalid_mask] = depth_max

        depth_scaled = (depth - depth_min) / (depth_max - depth_min)
        depth_scaled_uint8 = np.uint8(depth_scaled * 255)
        depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
        depth_color[invalid_mask.squeeze(), :] = 0

        return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
