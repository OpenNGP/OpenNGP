import numpy as np
import torch


if __name__ == '__main__':
    import gin
    from python_api.provider import get_dataset
    from python_api.provider.data_utils import cal_depth_confidences
    from python_api.app.config import Config
    from python_api.app.engine import Engine
    from os.path import exists, join as pjoin
    from os import makedirs
    from PIL import Image

    config_file = 'test/test_depth_consistency.gin'
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()
    if not exists(config.exp_dir): makedirs(config.exp_dir)

    config.factor = 7.5
    dataset = get_dataset('train', pjoin(config.data_dir, 'transforms_aligned.json') , config)
    img_dataset = dataset.dataset.img_dataset
    depths, masks = list(zip(*img_dataset.ori_depths))
    depths = np.stack(depths)
    masks = np.stack(masks)
    T = img_dataset.poses

    i_train = np.arange(len(T))
    K = np.identity(4, dtype=depths.dtype)
    K = K[None, ...].repeat(len(T), axis=0)
    K[:, :3, :3] = img_dataset.get_intrinsics(i_train)
    topk = 4
    confs = cal_depth_confidences(torch.from_numpy(depths),
                                  torch.from_numpy(T),
                                  torch.from_numpy(K),
                                  i_train,
                                  topk=topk)
    for idx, (depth, conf) in enumerate(zip(depths, confs)):
        img = Engine.visualize_depth(depth)
        Image.fromarray(img).save(pjoin(config.exp_dir, f'{idx:04d}_depth.png'))
        img = Engine.visualize_depth(conf, depth_min=0, depth_max=0.05, direct=True)
        Image.fromarray(img).save(pjoin(config.exp_dir, f'{idx:04d}_conf.png'))
    pass