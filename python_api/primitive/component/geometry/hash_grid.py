import torch
import torch.nn as nn
from python_api.primitive.component.module import encoder, regressor


class HashGrid(nn.Module):
    def __init__(self, bound, aabb) -> None:
        super(HashGrid, self).__init__()
        self.bound = bound
        self.encoder = encoder.HashEncoder(desired_resolution=2048)
        self.regressor = regressor.MLP(D=1,
                                       W=64,
                                       input_ch=self.encoder.output_dim,
                                       output_ch=1+15,
                                       skip_connections=[],
                                       bias=False)
        self.aabb = torch.Tensor(aabb)

    def forward(self, xyzs):
        inputs = self.encoder(xyzs, self.bound)
        outputs = self.regressor(inputs)  # 32->64-> 1+n
        sigmas = self.regressor.activation(outputs[..., 0])
        geo_features = outputs[..., 1:]
        return sigmas, geo_features
