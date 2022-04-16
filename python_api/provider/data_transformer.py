import torch.nn as nn
from typing import Union
from torch.utils.data import DataLoader
from python_api.provider.data_utils import prepare_data

from .dataset import Dataset


class DataTransformer(nn.Module):
    def __init__(self, dataset: Union[Dataset, DataLoader], device, config) -> None:
        """a handler to deal with possible data related learnable parameters
        exposure/intrinsics/extrinsics/etc

        Parameters
        ----------
        dataset : Union[Dataset, DataLoader]
            to query frame data (usually pixel and its corresponding ray)
        """
        self.dataset = dataset
        self.device = device


    def prepare_data(self, batch):
        return prepare_data(batch, self.device)
