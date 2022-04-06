from typing import Dict
from collections import namedtuple

from python_api.primitive.primitive import Primitive
from .integrator import integrator
from .raymarcher import raymarcher
from .rays import Rays

RenderPassResult = namedtuple(
    'RenderPassResult',
    ('xyzs',
     'views',
     'sigmas',
     'geo_features',
     'rgbs',
     'colors',))



class RenderPass:
    def __init__(self, sampler: str, integrator: str) -> None:
        self.sampler = sampler
        self.integrator = integrator

    def render_pixel(self, rays: Rays, primitive: Primitive, context: Dict):
        sampler_args = raymarcher.parameters(self.sampler)
        sampler_args = {context[k] for k in sampler_args}
        integrator_args = integrator.parameters(self.integrator)
        integrator_args = {context[k] for k in integrator_args}

        xyzs, views = raymarcher[self.sampler](rays, **sampler_args)
        sigmas, geo_features = primitive.query_sigma(xyzs)
        rgbs = primitive.query_color(geo_features, views)
        colors = integrator[self.integrator](sigmas, rgbs, **integrator_args)
        return RenderPassResult(xyzs,
                                views,
                                sigmas,
                                geo_features,
                                rgbs,
                                colors)
