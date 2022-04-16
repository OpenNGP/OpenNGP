from .dataset import Blender, LLFF, Multicam, Muyu
from .dataset_ngp import NeRFRayDataset
from .data_transformer import DataTransformer
from .data_utils import prepare_data



dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
    'muyu': Muyu,
    'NeRFRay': NeRFRayDataset
}


def get_dataset(split, train_dir, config):
    if 'train' == split and config.torch_dataset:
        # wrap it by data loader
        from torch.utils.data import DataLoader
        dataset = NeRFRayDataset('train', config.data_dir, config)
        dataset = DataLoader(dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=8)
    else:
        dataset = dataset_dict[config.dataset_loader](split, train_dir, config)
    return dataset


__all__ = [
    'get_dataset',
    'prepare_data',
    'DataTransformer'
]
