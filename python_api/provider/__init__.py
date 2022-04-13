from .dataset import Blender, LLFF, Multicam, Muyu
from .dataset_ngp import NeRFRayDataset


dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
    'muyu': Muyu,
    'NeRFRay': NeRFRayDataset
}


def get_dataset(split, train_dir, config):
  return dataset_dict[config.dataset_loader](split, train_dir, config)


__all__ = [
    'get_dataset'
]