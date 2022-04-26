import torch
import torch.nn as nn

from python_api.primitive.component.module import encoder, regressor


class Color(nn.Module):
    def __init__(self, N_emb, N_geo_emb, W) -> None:
        super(Color, self).__init__()
        self.encoder = encoder.Frequency(N_emb)
        self.regressor = regressor.MLP(D=0,
                                       W=W,
                                       input_ch=3*(1+2*N_emb)+N_geo_emb,  # encoded dirs + geo features
                                       output_ch=3,  # rgb
                                       skip_connections=[],
                                       act_on_last_layer=False)

    def forward(self, geo_features, views):
        view_inputs = self.encoder(views)
        color_inputs = torch.cat([geo_features, view_inputs], -1)
        outputs = self.regressor(color_inputs)
        return torch.sigmoid(outputs)
