from .component.appearance import color
from .component.geometry import sigma
from .primitive import Primitive


class BaseNeRF(Primitive):
    def __init__(self) -> None:
        self.appearance = color.Color()
        self.geometry = sigma.Sigma()
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return self.appearance.forward(geo_features, views)
