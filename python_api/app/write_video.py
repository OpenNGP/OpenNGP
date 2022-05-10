import gin
import numpy as np
from os import listdir, makedirs
from os.path import isfile, exists, join as pjoin
from PIL import Image
from python_api.app.config import Config


def main(config_file):
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()

    img_dir = pjoin(config.exp_dir, config.eval_dir)
    vid_dir = pjoin(img_dir, 'video')
    if not exists(vid_dir): makedirs(vid_dir)
    files = [pjoin(img_dir, f) for f in listdir(img_dir)]
    files = [f for f in files if isfile(f) and '.png' in f]

    rgb_files = sorted([f for f in files if '_depth' not in f])
    depth_files = sorted([f for f in files if '_depth' in f])

    rgbs = [np.array(Image.open(f)) for f in rgb_files]
    depths = [np.array(Image.open(f)) for f in depth_files]

    import imageio
    imageio.mimwrite(pjoin(vid_dir, 'rgb.mp4'), rgbs, fps=min(30, len(rgbs)), quality=8)
    imageio.mimwrite(pjoin(vid_dir, 'depth.mp4'), depths, fps=min(30, len(depths)), quality=8)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)