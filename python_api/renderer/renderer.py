from ..primitive.primitive import Primitive
from .integrator import Integrator
from .raymarcher import RayMarcher


class Renderer:
    def __init__(self) -> None:
        self.raymarcher = RayMarcher()
        self.integrator = Integrator()
        pass

    def render_color(self, rays_o, rays_d, primitive: Primitive):
        xyzs, views = self.raymarcher.sample_points(rays_o, rays_d)
        sigmas, geo_features = primitive.query_sigma(xyzs)
        rgbs = primitive.query_color(geo_features, views)
        return self.integrator.integrate(sigmas, rgbs)
