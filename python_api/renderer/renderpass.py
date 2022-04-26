from typing import Dict, Tuple
from collections import namedtuple

from python_api.primitive.primitive import Primitive
from .integrator import rayintegrator
from .raysampler import raysampler
from .rays import Rays

RenderPassResult = namedtuple(
    'RenderPassResult',
    ('name',
     'samples',
     'sigmas',
     'geo_features',
     'rgbs',
     'weights',
     'pixels'))



class RenderPass:
    def __init__(self,
                 name: str,
                 sampler: Tuple[str, Dict],
                 integrator: Tuple[str, Dict]) -> None:
        self.name = name

        self.sampler = sampler[0]
        self.sampler_args = raysampler.parameters(sampler[0])
        self.sampler_default_args = sampler[1]

        self.integrator = integrator[0]
        self.integrator_args = rayintegrator.parameters(integrator[0])
        self.integrator_default_args = integrator[1]

    def __repr__(self):
        return (f"[RenderPass]: {self.name}"
                f" sampler={self.sampler}"
                f" integrator={self.integrator}")

    def render_pixel(self, rays: Rays, primitive: Primitive, context: Dict):
        # render step
        context['primitive'] = primitive

        # 1. sample pts along ray
        sampler_ctx = {**self.sampler_default_args}
        sampler_ctx.update(context)
        sampler_ctx.update({'rays': rays})
        sampler_inputs = {k: sampler_ctx[k] for k in self.sampler_args}
        sampler_result = raysampler[self.sampler](**sampler_inputs)

        # 2. query infos from NGP
        sigmas, geo_features = primitive.query_sigma(sampler_result.xyzs)
        if 'rgbs' in self.integrator_args:  # save computation
            rgbs = primitive.query_color(geo_features, sampler_result.views)
        else:
            rgbs = None        

        # 3. integrate infos into pixel
        integrator_ctx = {**self.integrator_default_args}
        integrator_ctx.update(context)
        integrator_ctx.update(sampler_result._asdict())
        integrator_ctx.update({'sigmas': sigmas, 'rgbs': rgbs})
        intgr_inputs = {k: integrator_ctx[k] for k in self.integrator_args}
        weights, pixels = rayintegrator[self.integrator](**intgr_inputs)

        return RenderPassResult(self.name,
                                sampler_result,
                                sigmas,
                                geo_features,
                                rgbs,
                                weights,
                                pixels)
