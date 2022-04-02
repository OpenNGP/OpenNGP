from .component.appearance import color
from .component.geometry import sigma
from .primitive import Primitive


class BaseNeRF(Primitive):
    def __init__(self, multires=10, multires_views=4, netdepth=8, netwidth=256) -> None:
        self.appearance = color.Color(multires_views, netwidth, netwidth//2)
        self.geometry = sigma.Sigma(multires, netdepth, netwidth, [4])
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return self.appearance.forward(geo_features, views)
