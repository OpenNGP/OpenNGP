import torch
import torch.nn as nn
from python_api.primitive.component.module import encoder, regressor


class HashGrid(nn.Module):
    def __init__(self,
                 bound,
                 aabb,
                 num_levels, 
                 level_dim, 
                 base_resolution, 
                 log2_hashmap_size,
                 max_resolution,
                 N_geo_feature,
                 netdepth,
                 netwidth,
                 netbias,
                 skip_connections) -> None:
        super(HashGrid, self).__init__()
        self.bound = bound
        self.N_geo_feature = N_geo_feature
        self.encoder = encoder.HashEncoder(num_levels=num_levels,
                                           level_dim=level_dim,
                                           base_resolution=base_resolution,
                                           log2_hashmap_size=log2_hashmap_size,
                                           desired_resolution=max_resolution)
        self.regressor = regressor.MLP(D=netdepth,
                                       W=netwidth,
                                       input_ch=self.encoder.output_dim,
                                       output_ch=1+self.N_geo_feature,
                                       skip_connections=skip_connections,
                                       bias=netbias)
        self.aabb = torch.Tensor(aabb)

    def forward(self, xyzs):
        inputs = self.encoder(xyzs, self.bound)
        outputs = self.regressor(inputs)  # 32->64-> 1+n
        sigmas = self.regressor.activation(outputs[..., 0])
        geo_features = outputs[..., 1:]
        return sigmas, geo_features
