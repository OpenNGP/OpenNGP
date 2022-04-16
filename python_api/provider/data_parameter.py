"""Module for learnable per-data/dataset parameter

dataset-level instrinsic parameter
per-data exposuire
per-data pose

"""


import torch
import torch.nn as nn


class ExposureRefine(nn.Module):
    def __init__(self, num_frame):
        super().__init__()

        self.num_frame = num_frame
        self.num_param = 3  # white balance
        self.vars = nn.Parameter(torch.zeros(self.num_frame, self.num_param))

    def get_exposure_scale(self, ids):
        expos = torch.exp(0.6931471805599453*self.vars[ids])
        return expos.squeeze(2)
