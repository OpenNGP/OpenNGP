from turtle import forward
from ..module import encoder, regressor


class Color:
    def __init__(self) -> None:
        self.encoder = encoder.Frequency()
        self.regressor = regressor.Linear()

    def forward(self, geo_features, views):
        pass
