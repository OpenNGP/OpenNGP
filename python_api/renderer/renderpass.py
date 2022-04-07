from typing import Dict
from collections import namedtuple

from python_api.primitive.primitive import Primitive
from .integrator import rayintegrator
from .raymarcher import raymarcher
from .rays import Rays

RenderPassResult = namedtuple(
    'RenderPassResult',
    ('xyzs',
     'views',
     'sigmas',
     'geo_features',
     'rgbs',
     'weights',
     'pixels',))



class RenderPass:
    def __init__(self, sampler: str, integrator: str) -> None:
        self.sampler = raymarcher[self.sampler]
        self.sampler_args = raymarcher.parameters(sampler)
        self.integrator = rayintegrator[self.integrator]
        self.integrator_args = rayintegrator.parameters(integrator)

    def render_pixel(self, rays: Rays, primitive: Primitive, context: Dict):
        sampler_args = {context[k] for k in self.sampler_args}
        integrator_args = {context[k] for k in self.integrator_args}

        xyzs, views = self.sampler(rays, **sampler_args)
        sigmas, geo_features = primitive.query_sigma(xyzs)
        rgbs = primitive.query_color(geo_features, views)
        weights, pixels = self.integrator(sigmas, rgbs, **integrator_args)
        return RenderPassResult(xyzs,
                                views,
                                sigmas,
                                geo_features,
                                rgbs,
                                weights,
                                pixels)
