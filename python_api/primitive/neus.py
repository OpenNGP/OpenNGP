from .primitive import Primitive
from .component.geometry.sdf import SDFNet
from .component.appearance.color import Color

class NeuS(Primitive):
    def __init__(self) -> None:
        self.geometry = SDFNet(d_in=3, d_out=257, d_hidden=256, n_layers=8)
        self.appearance = Color()
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return super().query_color(geo_features, views)
