from .component.appearance import Color, ColorSH
from .component.geometry import MockHashGrid, HashGrid
from .primitive import Primitive


class InstantNGPMock(Primitive):
    def __init__(self, bound, aabb=None) -> None:
        if aabb is None:
            aabb = [-bound, -bound, -bound, bound, bound, bound]
        self.geometry = MockHashGrid(bound, aabb)
        N_geo_feature = self.geometry.N_geo_feature
        self.appearance = Color(4, N_geo_feature, N_geo_feature//2)
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


class InstantNGPMock2(Primitive):
    def __init__(self, bound, aabb=None) -> None:
        if aabb is None:
            aabb = [-bound, -bound, -bound, bound, bound, bound]
        # self.geometry = MockHashGrid(bound, aabb)
        self.geometry = HashGrid(bound=bound,
                                 aabb=aabb,
                                 num_levels=1,
                                 level_dim=64,
                                 base_resolution=128,
                                 log2_hashmap_size=21,
                                 max_resolution=None,
                                 N_geo_feature=256,
                                 netdepth=1,
                                 netwidth=256,
                                 netbias=True,
                                 skip_connections=[])
        N_geo_feature = self.geometry.N_geo_feature
        self.appearance = Color(4, N_geo_feature, N_geo_feature//2)
        # self.appearance = ColorSH(4, N_geo_feature, 2, 64)
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
