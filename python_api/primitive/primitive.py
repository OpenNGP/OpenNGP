import numpy as np
from abc import ABC


class Primitive(ABC):
    def __init__(self) -> None:
        super().__init__()

    def default_query_ctx(self):
        return {}

    def default_query_sigma_ctx(self):
        return {}

    def default_query_color_ctx(self):
        return {}

    def query(self, xyzs):
        return np.array([])

    def query_sigma(self, xyzs):
        return np.array([])

    def query_color(self, geo_features, views, **args):
        return np.array([])

    def query_color(self, geo_features, xyzs, views, **args):
        return np.array([])

    def build_module_graph(self, module_key, module):
        pass

    def to(self, device):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def load(self, state_dict):
        raise NotImplementedError
