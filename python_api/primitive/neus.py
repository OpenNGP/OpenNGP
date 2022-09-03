from .primitive import Primitive
from .component.geometry.sdf import SDFNet
from .component.appearance.color import NeusColor

class NeuS(Primitive):
    def __init__(self) -> None:
        ### TODO add parameter in config
        self.geometry = SDFNet(d_in=3, d_out=257, d_hidden=256, n_layers=8, 
                                skip_in=[4], multires=6)
        self.appearance = NeusColor(d_feature=256, d_in=9, d_out=3, d_hidden=256,
                            n_layers=4, multires_view=4, squeeze_out=True)
        pass

    def query_sigma(self, xyzs):
        return self.geometry.forward(xyzs)

    def query_color(self, geo_features, views):
        return super().query_color(geo_features, views)

    def query_color(self, points, normals, view_dirs, feature_vectors):
        return self.appearance.forward(points, normals, view_dirs, feature_vectors)
