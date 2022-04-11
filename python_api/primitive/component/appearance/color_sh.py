import torch
import torch.nn as nn

from python_api.primitive.component.module import encoder, regressor


class ColorSH(nn.Module):
    def __init__(self, N_degree, N_geo_emb, W) -> None:
        super(ColorSH, self).__init__()
        self.encoder = encoder.SphericalHarmonic(N_degree)
        self.regressor = regressor.MLP(D=0,
                                       W=W,
                                       input_ch=self.encoder.output_ch+N_geo_emb,  # encoded dirs + geo features
                                       output_ch=3,  # rgb
                                       skip_connections=[],
                                       act_on_last_layer=False)

    def forward(self, geo_features, views):
        view_inputs = self.encoder(views)
        color_inputs = torch.cat([geo_features, view_inputs], -1)
        outputs = self.regressor(color_inputs)
        return torch.sigmoid(outputs)
