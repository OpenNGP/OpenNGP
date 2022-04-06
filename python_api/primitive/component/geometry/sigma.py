from python_api.primitive.component.module import encoder, regressor


class Sigma:
    def __init__(self, N_emb, D, W, skip_connections) -> None:
        
        self.encoder = encoder.Frequency(N_emb)
        self.regressor = regressor.MLP(D=D,
                                       W=W,
                                       input_ch=3*(1+N_emb),  # encoded xyzs
                                       output_ch=1+W,
                                       skip_connections=skip_connections,
                                       act_on_last_layer=False)

    def forward(self, xyzs):
        inputs = self.encoder(xyzs)
        outputs = self.regressor(inputs)
        sigmas = self.regressor.activation(outputs[:, 0])
        geo_features = outputs[:, 1:]
        return sigmas, geo_features
