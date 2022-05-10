from .component.appearance import ColorSH
from .component.geometry import HashGrid
from .primitive import Primitive


class InstantNGP(Primitive):
    def __init__(self, bound, aabb=None) -> None:
        if aabb is None:
            aabb = [-bound, -bound, -bound, bound, bound, bound]
        self.geometry = HashGrid(bound=bound,
                                 aabb=aabb,
                                 num_levels=16,
                                 level_dim=2,
                                 base_resolution=16,
                                 log2_hashmap_size=19,
                                 max_resolution=2048,
                                 N_geo_feature=15,
                                 netdepth=1,
                                 netwidth=64,
                                 netbias=False,
                                 skip_connections=[])
        N_geo_feature = self.geometry.N_geo_feature
        self.appearance = ColorSH(4, N_geo_feature, 2, 64)
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return self.appearance.forward(geo_features, views)

    def to(self, device):
        self.appearance.to(device)
        self.geometry.to(device)
        self.geometry.aabb = self.geometry.aabb.to(device)

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
