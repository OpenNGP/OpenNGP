import torch
import torch.nn as nn
from python_api.primitive.component.module import encoder, regressor


class MockHashGrid(nn.Module):
    def __init__(self, bound, aabb) -> None:
        super(MockHashGrid, self).__init__()
        self.bound = bound
        self.N_geo_feature = 256
        N_freq_band = 10
        self.encoder = encoder.Frequency(N_freq_band)
        self.regressor = regressor.MLP(D=8,
                                       W=256,
                                       input_ch=3*(1+2*N_freq_band),
                                       output_ch=1+self.N_geo_feature,
                                       skip_connections=[4],
                                       bias=True)
        self.aabb = torch.Tensor(aabb)

    def forward(self, xyzs):
        inputs = self.encoder(xyzs)
        outputs = self.regressor(inputs)  # 32->64-> 1+n
        sigmas = self.regressor.activation(outputs[..., 0])
        geo_features = outputs[..., 1:]
        return sigmas, geo_features
