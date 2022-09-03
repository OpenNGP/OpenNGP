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


class NeusColor(nn.Module):
    """
        Besides viewdir, this net also takes point position and normal as input
    """
    def __init__(self, d_feature, d_in, d_out, d_hidden, n_layers, multires_view, squeeze_out=True) -> None:
        super(NeusColor, self).__init__()
        self.squeeze_out = squeeze_out
        self.encoder = encoder.Frequency(multires_view)
        input_ch = 3*2*multires_view + d_in + d_feature  # position encoding for viewdir, d_in may contain position and normal
        self.regressor = regressor.MLP(D=n_layers,
                                       W=d_hidden,
                                       input_ch=input_ch,
                                       output_ch=d_out,
                                       activation=torch.nn.functional.relu,
                                       act_on_last_layer=False
                                       )

    def forward(self, points, normals, view_dirs, feature_vectors):
        view_inputs = self.encoder(view_dirs)
        rendering_input = torch.cat([points, view_inputs, normals, feature_vectors], dim=-1)
        outputs = self.regressor(rendering_input)
        if self.squeeze_out:
            outputs = torch.sigmoid(outputs)
        return outputs