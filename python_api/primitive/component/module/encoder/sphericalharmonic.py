import torch
from torch import nn


def _degree_1(x):
    return [0.28209479177387814 * torch.ones_like(x)]


def _degree_2(x, y, z):
    return [
        -0.48860251190291987*y,
        0.48860251190291987*z,
        -0.48860251190291987*x,
    ]


def _degree_3(xy, xz, yz, x2, y2, z2):
    return [
        1.0925484305920792*xy,
        -1.0925484305920792*yz,
        0.94617469575755997*z2 - 0.31539156525251999,
        -1.0925484305920792*xz,
        0.54627421529603959*x2 - 0.54627421529603959*y2,
    ]


def _degree_4(x, y, z, x2, y2, z2, xy):
    return [
        0.59004358992664352*y*(-3.0*x2 + y2),
        2.8906114426405538*xy*z,
        0.45704579946446572*y*(1.0 - 5.0*z2),
        0.3731763325901154*z*(5.0*z2 - 3.0),
        0.45704579946446572*x*(1.0 - 5.0*z2),
        1.4453057213202769*z*(x2 - y2),
        0.59004358992664352*x*(-x2 + 3.0*y2),
    ]


def _degree_5(x2, y2, z2, xy, xz, yz, x4, y4, z4):
    return [
        2.5033429417967046*xy*(x2 - y2),
        1.7701307697799304*yz*(-3.0*x2 + y2),
        0.94617469575756008*xy*(7.0*z2 - 1.0),
        0.66904654355728921*yz*(3.0 - 7.0*z2),
        -3.1735664074561294*z2 + 3.7024941420321507*z4 + 0.31735664074561293,
        0.66904654355728921*xz*(3.0 - 7.0*z2),
        0.47308734787878004*(x2 - y2)*(7.0*z2 - 1.0),
        1.7701307697799304*xz*(-x2 + 3.0*y2),
        -3.7550144126950569*x2*y2 + 0.62583573544917614*x4 + 0.62583573544917614*y4,
    ]


class SphericalHarmonic(nn.Module):
    def __init__(self, N_degrees) -> None:
        super(SphericalHarmonic, self).__init__()
        self.N_degrees = N_degrees

    @property
    def output_ch(self):
        return (self.N_degrees)**2

    def forward(self, inputs):
        x, y, z = inputs[..., 0:1], inputs[..., 1:2], inputs[..., 2:3]

        outputs = []
        if self.N_degrees <= 1:
            outputs += _degree_1(x)
        if self.N_degrees <= 2:
            outputs += _degree_1(x) + _degree_2(x, y, z)
        elif self.N_degrees <= 3:
            xy, xz, yz = x*y, x*z, y*z
            x2, y2, z2 = x*x, y*y, z*z
            outputs += (_degree_1(x) +
                        _degree_2(x, y, z) +
                        _degree_3(xy, xz, yz, x2, y2, z2))
        elif self.N_degrees <= 4:
            xy, xz, yz = x*y, x*z, y*z
            x2, y2, z2 = x*x, y*y, z*z
            # xyz = xy*z
            outputs += (_degree_1(x) +
                        _degree_2(x, y, z) +
                        _degree_3(xy, xz, yz, x2, y2, z2) +
                        _degree_4(x, y, z, x2, y2, z2, xy))
        elif self.N_degrees <= 5:
            xy, xz, yz = x*y, x*z, y*z
            x2, y2, z2 = x*x, y*y, z*z
            x4, y4, z4 = x2*x2, y2*y2, z2*z2
            outputs += (_degree_1(x) +
                        _degree_2(x, y, z) +
                        _degree_3(xy, xz, yz, x2, y2, z2) +
                        _degree_4(x, y, z, x2, y2, z2, xy) +
                        _degree_5(x2, y2, z2, xy, xz, yz, x4, y4, z4))
        return torch.concat(outputs, -1)
