import numpy as np
from abc import ABC


class Primitive(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    def query(xyzs):
        return np.array([])

    def query_sigma(xyzs):
        return np.array([])

    def query_color(geo_features, views):
        return np.array([])

    def build_module_graph(module_key, module):
        pass
