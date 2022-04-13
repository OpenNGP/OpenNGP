from .component.appearance import ColorSH
from .component.geometry import HashGrid
from .primitive import Primitive


class InstantNGP(Primitive):
    def __init__(self, bound) -> None:
        self.geometry = HashGrid(bound)
        self.appearance = ColorSH(4, 15, 2, 64)
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return self.appearance.forward(geo_features, views)

    def to(self, device):
        self.appearance.to(device)
        self.geometry.to(device)

    def export(self):
        return {
            'geometry': self.geometry.state_dict(),
            'appearance': self.appearance.state_dict()
        }

    def parameters(self):
        return list(self.geometry.parameters()) + list(self.appearance.parameters())

    def load(self, state_dict):
        self.geometry.load_state_dict(state_dict['geometry'])
        self.appearance.load_state_dict(state_dict['appearance'])
