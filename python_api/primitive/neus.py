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

    def default_query_sigma_ctx(self):
        return {'normal': False}

    def default_query_color_ctx(self):
        return {'xyzs': True}

    def to(self, device):
        self.appearance.to(device)
        self.geometry.to(device)
        self.deviation.to(device)

    def parameters(self):
        return list(self.geometry.parameters()) + list(self.appearance.parameters()) +\
            list(self.deviation.parameters())

    # def query_sigma(self, xyzs):
    #     sdf, geo_features = self.geometry.forward(xyzs)
    #     # self.normal_cache = None
    #     # return sdf, torch.cat([xyzs, geo_features], dim=1)
    #     return sdf, geo_features  # [N_ray,N_sample,1], [N_ray,N_sample,3]

    def query_sigma(self, xyzs, normal=False):
        sdf, geo_features = self.geometry.forward(xyzs)
        if normal:
            normal = self.geometry.gradient(xyzs)
            geo_features = torch.cat([normal, geo_features], dim=-1)
        return sdf, geo_features


    # def query_color(self, points, normals, view_dirs, feature_vectors):
    #     return self.appearance.forward(points, normals, view_dirs, feature_vectors)

    def query_normal(self, points):
        return self.geometry.gradient(points)

    def query_color(self, geo_features, views, xyzs):
        normals, feature_vectors = geo_features.split([3,self.d_feature], dim=-1)
        return self.appearance.forward(xyzs, normals, views, feature_vectors)

    def query_shaped_invs(self, shape):
        """
            query inv_s with given shape
        """
        return self.deviation.forward(shape)