import torch
from torch import nn
import torch.nn.functional as F
from python_api.primitive.component.module import encoder, regressor

class SDFNet(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), N_emb=0, bias=0.5, \
                    geometric_init=True, weight_norm=True, inside_outside=False):
        """

        Args:
            d_in (int): input dimension, e.g., 3
            d_out (int): output dimension, e.g., 257
            d_hidden (int): MLP hidden dimension, e.g., 256 
            n_layers (int): lays of MLP, e.g., 8
            skip_in (tuple, optional): Defaults to (4,).
            N_emb (int, optional): Dimension of positional encoding. Defaults to 0.
            bias (float, optional): ****** Defaults to 0.5.
            geometric_init (bool, optional): ******. Defaults to True.
            weight_norm (bool, optional): ******. Defaults to True.
            inside_outside (bool, optional): ******. Defaults to False.
        """
        super().__init__()
        self.encoder = encoder.Frequency(N_emb)
        d_in = d_in * (1 + 2*N_emb)  # encoded xyzs
        activation = lambda x: F.softplus(x, beta=200)
        self.regressor = regressor.MLP(D=n_layers,
                                       W=d_hidden,
                                       input_ch=d_in*(1+2*N_emb),  # encoded xyzs
                                       output_ch=d_out,
                                       skip_connections=skip_in,
                                       act_on_last_layer=False,
                                       activation=activation)
        # TODO: add the parameter initial part 


    def forward(self, xyzs):
        inputs = self.encoder(xyzs)
        outputs = self.regressor(inputs)
        sdf = outputs[..., 0]
        geo_features = outputs[..., 1:]
        return sdf, geo_features


    def gradient(self, xyzs):
        xyzs.requires_grad_(True)
        y = self.forward(xyzs)[0]  # query the sdf
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xyzs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    
