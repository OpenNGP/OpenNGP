import numpy as np
from abc import ABC


class Primitive(ABC):
    def __init__(self) -> None:
        super().__init__()

    def query(self, xyzs):
        return np.array([])

    def query_sigma(self, xyzs):
        return np.array([])

    def query_color(self, geo_features, views):
        return np.array([])

    def build_module_graph(self, module_key, module):
        pass

    def to(self, device):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError
