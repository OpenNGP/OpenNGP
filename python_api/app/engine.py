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
        self.pipelines = [self.build_pipeline(p) for p in config.pipelines]

    def build_pipeline(self, pipeline_def):
        name = pipeline_def['name']
        ngp: Primitive = NGP_factory.build(**pipeline_def['arch'])
        ngp.to(self.device)
        renderer = Renderer(pipeline_def['renderpasses'])
        self.console.print(f'\[{name}] NGP Geometry: ', ngp.geometry)
        self.console.print(f'\[{name}] NGP coarse Appearance: ', ngp.appearance)
        self.console.print(f'\[{name}] Renderer: ', renderer)
        return Pipeline(name, ngp, renderer)

    def collect_parameters(self):
        params = []
        for pipeline in self.pipelines:
            params += pipeline.ngp.parameters()
        return params

    def load_ngp(self, state_dict):
        [p.ngp.load(state_dict[p.name]) for p in self.pipelines]

    def export_ngp(self):
        return {p.name: p.ngp.export() for p in self.pipelines}

    def run(self, rays, context={}):
        rets, ctx = [], context
        for pipeline in self.pipelines:
            rets += pipeline.renderer.render(rays, pipeline.ngp, ctx)
            ctx.update(rets[-1]._asdict())
        return rets

    def prepare_data(self, batch):
        device = self.device
        if isinstance(batch['rays'].origins, np.ndarray):
            map_func = lambda r: torch.from_numpy(r).to(device, non_blocking=True)
        else:
            map_func = lambda r: r.to(device, non_blocking=True)

        for k, v in batch.items():
            if isinstance(v, namedtuple):
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
