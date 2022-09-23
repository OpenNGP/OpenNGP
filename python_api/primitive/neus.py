import torch
from .primitive import Primitive
from .component.geometry.sdf import SDFNet
from .component.appearance.color import NeusColor
from .component.module.learnableParam import SingleVariable

class NeuS(Primitive):
    def __init__(self) -> None:
        ### TODO add parameter in config
        self.d_feature = 256
        self.geometry = SDFNet(d_in=3, d_out=self.d_feature+1, d_hidden=256, n_layers=8, 
                                skip_in=[4], multires=6)
        self.appearance = NeusColor(d_feature=self.d_feature, d_in=9, d_out=3, d_hidden=256,
                            n_layers=4, multires_view=4, squeeze_out=True)
        self.deviation = SingleVariable(init_val=0.3)

        self.normal_cache = None
        pass

    def query_sigma(self, xyzs):
        sdf, geo_features = self.geometry.forward(xyzs)
        self.normal_cache = None
        return sdf, torch.cat([xyzs, geo_features], dim=1)

    def query_color(self, points, normals, view_dirs, feature_vectors):
        return self.appearance.forward(points, normals, view_dirs, feature_vectors)

    def query_color(self, geo_features, view_dirs):
        points, feature_vectors = geo_features.split([3,self.d_feature,3], dim=1)
        normals = self.geometry.gradient(points)
        self.normal_cache = normals 
        return self.appearance.forward(points, normals, view_dirs, feature_vectors)

    def query_shaped_invs(self, shape):
        """
            query inv_s with given shape
        """
        return self.deviation.forward(shape)