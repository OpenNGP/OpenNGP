import torch
import numpy as np

from python_api.utils.data_helper import namedtuple_map


def prepare_data(batch, device):
    if isinstance(batch['rays'].origins, np.ndarray):
        def map_func(r): return torch.from_numpy(
            r).to(device, non_blocking=True)
    else:
        def map_func(r): return r.to(device, non_blocking=True)

    for k, v in batch.items():
        if isinstance(v, tuple):
            batch[k] = namedtuple_map(map_func, v)
        else:
            batch[k] = map_func(v)
    return batch
