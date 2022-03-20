from ..module import encoder, regressor


class Sigma:
    def __init__(self, ) -> None:
        self.encoder = encoder.Frequency()
        self.regressor = regressor.Linear()

    def forward(self, xyzs):
        pass
