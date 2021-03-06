from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, D=8,
                 W=256,
                 input_ch=3,
                 output_ch=4,
                 skip_connections=[4],
                 activation=F.relu,
                 act_on_last_layer=False,
                 bias=True) -> None:
        super(MLP, self).__init__()

        self.skip_connections = skip_connections
        first_layer = [nn.Linear(input_ch, W, bias)]
        hidden_layers = [nn.Linear(W, W, bias) if i not in self.skip_connections
                         else nn.Linear(W + input_ch, W, bias)
                         for i in range(D-1)]
        self.linears = nn.ModuleList(first_layer+hidden_layers)
        self.activation = activation
        self.last_layer = nn.Linear(W, output_ch, bias)
        self.act_on_last_layer = act_on_last_layer
        pass

    def forward(self, x):
        h = x
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.activation(h)
            if i in self.skip_connections:
                h = torch.cat([x, h], -1)
        if self.act_on_last_layer:
            return self.activation(self.last_layer(h))
        else:
            return self.last_layer(h)
