import gin
import numpy as np
import torch
import tqdm

from os.path import exists, join as pjoin
from os import makedirs
from PIL import Image

from python_api.app.engine import Engine
from python_api.app.config import Config

from python_api.provider import get_dataset
from python_api.utils.data_helper import namedtuple_map


def save_checkpoint(state_dict, save_path, device):
    if exists(save_path):
        last_state_dict = torch.load(save_path, map_location=device)
        if last_state_dict['best_metric'] > state_dict['best_metric']:
            return
    torch.save(state_dict, save_path)


def main(config_file):
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()

    print('==> build NGP and renderer')
    engine = Engine(config)
    device = engine.device

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print('==> build dataset')
    config.render_path = True
    config.batch_size = 2048
    val_dataset = get_dataset('test', config.data_dir, config)

    eval_save_dir = pjoin(config.exp_dir, "eval")
    if not exists(eval_save_dir): makedirs(eval_save_dir)

    best_ckpt = pjoin(config.exp_dir, 'ckpt_best.pth.tar')
    if exists(best_ckpt):
        state_dict = torch.load(best_ckpt, map_location=device)
        engine.load_ngp(state_dict)
        init_step = 1
    else:
        raise ValueError

    pbar = tqdm.tqdm(total=val_dataset.size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    pbar.update(init_step-1)

    print('==> start evaluating NGP')
    for step, test_batch in zip(range(init_step, val_dataset.size + 1), val_dataset):
        chunk = val_dataset.batch_size
        test_batch = engine.prepare_data(test_batch)
        test_rays = test_batch['rays']
        height, width = test_rays[0].shape[:2]
        num_rays = height * width
        test_rays = namedtuple_map(
            lambda r: r.reshape((num_rays, -1)),
            test_rays
        )
        results = []
        with torch.no_grad():
            for i in range(0, num_rays, chunk):
                # pylint: disable=cell-var-from-loop
                chunk_rays = namedtuple_map(
                    lambda r: r[i:i + chunk],
                    test_rays
                )
                rets_test = engine.run_eval(chunk_rays, {'perturb': False})
                results.append(rets_test[-1].pixels.colors)
            test_pixels = torch.concat(results)
            test_pixels = test_pixels.reshape((height, width, -1))
            img = (test_pixels.detach().cpu().numpy() * 255).astype(np.uint8)
        
        # path_depth = pjoin(eval_save_dir, f'{i:04d}_depth.png')
        Image.fromarray(img).save(pjoin(eval_save_dir, f'{step-1:04d}.png'))
        pbar.update()

    print('==> end evaluating NGP')
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)
