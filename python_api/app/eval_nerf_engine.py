import gin
import numpy as np
import torch
import tqdm

from os.path import exists, join as pjoin
from os import makedirs
from PIL import Image

from python_api.app.engine import Engine
from python_api.app.config import Config

from python_api.provider import get_dataset, prepare_data


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

    eval_save_dir = pjoin(config.exp_dir, config.eval_dir)
    if not exists(eval_save_dir): makedirs(eval_save_dir)

    best_ckpt = pjoin(config.exp_dir, 'ckpt_best.pth.tar')
    if exists(best_ckpt):
        state_dict = torch.load(best_ckpt, map_location=device)
        engine.load_ngp(state_dict)
        init_step = 1
    else:
        raise ValueError

    print('==> build dataset')
    config.render_path = True
    config.batch_size = 2048
    val_dataset = get_dataset('test', config.data_dir, config)

    pbar = tqdm.tqdm(total=val_dataset.size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    pbar.update(init_step-1)

    print('==> start evaluating NGP')
    for step, test_batch in zip(range(init_step, val_dataset.size + 1), val_dataset):
        test_batch = prepare_data(test_batch, engine.device)
        rgb, depth = engine.draw(test_batch['rays'], val_dataset.batch_size)

        rgb = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(pjoin(eval_save_dir, f'{step-1:04d}.png'))

        depth = engine.visualize_depth(depth.cpu().numpy())
        Image.fromarray(depth).save(pjoin(eval_save_dir, f'{step-1:04d}_depth.png'))
        pbar.update()

    print('==> end evaluating NGP')
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)
