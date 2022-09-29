import gin
import torch
import numpy as np
import json

from os.path import exists, join as pjoin, splitext, basename
from os import makedirs
from matplotlib import cm
from trimesh import Trimesh
from trimesh.exchange.ply import export_ply
from PIL import Image
from python_api.app.config import Config
from python_api.app.engine import Engine
from python_api.provider import get_dataset
from python_api.provider.data_utils import est_global_scale


def analyze_coverage(frame_pts, bound, grid_res):
    grid_resolution = grid_res
    coverage_stat = []
    all_entries = set()
    for frame_idx, inputs in enumerate(frame_pts):
        inputs = (inputs + bound) / (2 * bound)  # map to [0, 1]
        H = int(np.ceil(grid_resolution + 1)) + 1  # +1 for voxel corner
        voxel_offset = np.array([[i >> 2 & 1, i >> 1 & 1, i >> 0 & 1] for i in range(8)])
        pos = inputs * grid_resolution + 0.5
        pos_grid = np.floor(pos)
        pos_grid = pos_grid[:, 0]*H*H + pos_grid[:, 1]*H + pos_grid[:, 2]
        pos_grid, counts = np.unique(pos_grid.astype(int), return_counts=True)
        cur_entry = set(pos_grid)
        all_entries.update(cur_entry)
        coverage_stat.append({'frame_idx': frame_idx,
                              'pos': cur_entry,
                              'counts': counts})

    # coverage_stat = sorted(coverage_stat, key=lambda item: len(item['pos']))
    best_frames = []
    while len(all_entries) > 0:
        adj_weights = np.zeros((len(frame_pts), len(frame_pts)), dtype=int)        
        for i in range(len(frame_pts)):
            pos_i = coverage_stat[i]['pos']
            adj_weights[i, i] = len(all_entries.intersection(pos_i))
            for j in range(i+1, len(frame_pts)):
                pos_j = coverage_stat[j]['pos']
                w = len(pos_i.intersection(pos_j))
                adj_weights[i, j] = w
                adj_weights[j, i] = w

        diag = adj_weights.diagonal()
        # w_sum = adj_weights.sum(axis=1)
        w_sum = diag
        best_i = np.argmax(w_sum)
        if 0 == w_sum[best_i]:
            best_i = np.argmax(diag)
        pos_i = coverage_stat[best_i]['pos']
        count_before = len(all_entries)
        all_entries.difference_update(pos_i)
        for update_i in range(len(frame_pts)):
            coverage_stat[update_i]['pos'].difference_update(pos_i)
        count_after = len(all_entries)
        count_this_frame = count_before-count_after
        if count_this_frame >= 50:
            best_frames.append(best_i)
        print(f'entries before: {count_before}, entries after: {count_after}, cover_in_this_frame: {count_this_frame}')
        pass
    return best_frames


def main(config_file):
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    config.dataset_loader = 'NeRFTestIter'
    config.factor = 7.5
    dataset = get_dataset('test_train', pjoin(config.data_dir, 'transforms_aligned.json') , config)
    datas = []
    debug_dir = pjoin(config.exp_dir, 'debug')
    if not exists(debug_dir): makedirs(debug_dir)
    for idx in range(dataset.size):
        # data = dataset.img_dataset[idx]
        # datas.append(data)
        # name = dataset.img_dataset.names[idx]
        # depth = dataset.img_dataset.depths[idx]
        # mask = dataset.img_dataset.masks[idx]
        # depth = Engine.visualize_depth(depth, mask)
        # Image.fromarray(depth).save(pjoin(debug_dir, f'{name}_depth.png'))

        pass

    # dataset = get_dataset('train', config.data_dir, config)
    # tdataset = get_dataset('test', config.data_dir, config)
    # next(tdataset)
    color_frame_pts = dataset.img_dataset.export_pointcloud(pjoin(debug_dir, 'pointcloud'))
    frame_pts = dataset.img_dataset.depth_pointcloud()
    scale, offset = est_global_scale(frame_pts)
    inputs = np.vstack(frame_pts)
    bound = config.primitives[0]['arch']['bound']
    inputs = (inputs + bound) / (2 * bound)  # map to [0, 1]
    grid_resolution = 64

    best_frames = analyze_coverage(frame_pts, bound, grid_resolution)
    best_frames_name = [dataset.img_dataset[frame_idx]['name'] for frame_idx in best_frames]
    
    transforms = json.load(open(pjoin(config.data_dir, 'transforms_aligned.json')))
    frames = []
    redundant_frames = []
    for frame in transforms['frames']:
        name = splitext(basename(frame['file_path']))
        if name[0] in best_frames_name:
            frames.append(frame)
        else:
            redundant_frames.append(frame)
    # insert a redundant frames into middle of training set
    mid = (len(frames)+1)//2
    frames = frames[:mid] + [redundant_frames[0]] + frames[mid:]
    transforms['frames'] = frames
    json.dump(transforms,
              open(pjoin(config.data_dir,
                         'transforms_aligned_best_coverage.json'),
                   'w'),
              indent=2)
    
    cov_output_dir = pjoin(debug_dir, 'pointcloud_best_coverage')
    if not exists(cov_output_dir): makedirs(cov_output_dir)
    for frame_idx in best_frames:
        name = dataset.img_dataset[frame_idx]['name']
        pts = color_frame_pts[frame_idx]
        mesh = Trimesh(vertices=pts[:, :3], vertex_colors=pts[:, 3:], process=False)
        open(pjoin(cov_output_dir, f'{name}.ply'), 'wb').write(export_ply(mesh))


    H = int(np.ceil(grid_resolution + 1)) + 1  # +1 for voxel corner
    voxel_offset = np.array([[i >> 2 & 1, i >> 1 & 1, i >> 0 & 1] for i in range(8)])
    pos = inputs * grid_resolution + 0.5
    pos_grid = np.floor(pos)
    pos_grid = pos_grid[:, 0]*H*H + pos_grid[:, 1]*H + pos_grid[:, 2]
    pos_grid_bkp = pos_grid
    pos_grid, counts = np.unique(pos_grid.astype(int), return_counts=True)

    coverage = cm.get_cmap('jet')(counts/counts.max())
    coverage = (coverage[..., :3]*255).astype(np.uint8)

    pos_i = pos_grid//(H*H)
    pos_j = (pos_grid-pos_i*H*H)//H
    pos_k = pos_grid-pos_i*H*H-pos_j*H
    xyz = np.concatenate([pos_i, pos_j, pos_k])
    xyz = xyz.reshape((3, -1)).T
    xyz = (xyz + 0.5) / grid_resolution * (2*bound) - bound

    mesh = Trimesh(vertices=xyz, vertex_colors=coverage, process=False)
    open(pjoin(config.exp_dir, f'coverage_{grid_resolution}.ply'), 'wb').write(export_ply(mesh))

    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)