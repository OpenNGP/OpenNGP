from .component.appearance import color_sh
from .component.geometry import sigma
from .primitive import Primitive


class BaseNeRFSH(Primitive):
    def __init__(self, multires=10, multires_views=4, netdepth=8, netwidth=256) -> None:
        self.geometry = sigma.Sigma(multires, netdepth, netwidth, multires_views**2, [4])
        self.appearance = color_sh.ColorSH(multires_views, multires_views**2, 0, netwidth//2)
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return self.appearance.forward(geo_features, views)

    def to(self, device):
        self.appearance.to(device)
        self.geometry.to(device)

    def parameters(self):
        return list(self.geometry.parameters()) + list(self.appearance.parameters())
