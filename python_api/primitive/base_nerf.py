from .component.appearance import color
from .component.geometry import sigma
from .primitive import Primitive


class BaseNeRF(Primitive):
    def __init__(self, multires=10, multires_views=4, netdepth=8, netwidth=256) -> None:
        self.geometry = sigma.Sigma(multires, netdepth, netwidth, netwidth, [4])
        self.appearance = color.Color(multires_views, netwidth, netwidth//2)
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views, **args):
        return self.appearance.forward(geo_features, views)

    def to(self, device):
        self.appearance.to(device)
        self.geometry.to(device)

    def parameters(self):
        return list(self.geometry.parameters()) + list(self.appearance.parameters())

    def export(self):
        return {
            'geometry': self.geometry.state_dict(),
            'appearance': self.appearance.state_dict()
        }
    
    def load(self, state_dict):
        self.geometry.load_state_dict(state_dict['geometry'])
        self.appearance.load_state_dict(state_dict['appearance'])
