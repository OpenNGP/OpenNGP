import json
from shutil import copyfileobj
import pandas as pd
import requests
import numpy as np
from scipy.interpolate import UnivariateSpline, splev, BSpline
from scipy.spatial.transform import Slerp, Rotation
from os.path import basename, splitext, join as pjoin
from os.path import dirname, exists
from os import makedirs
from copy import deepcopy
from tqdm import tqdm
from ff3d_utils import OssManager
from trimesh import Trimesh
from trimesh.exchange.ply import export_ply


MOCK = {
        "bizId": "muyu-0-depthPlane",
        "camera": {},
        "imageName": "muyu-0-depthPlane",
        "renderEngineType": "AceRay",
        "height": 1080,
        "width": 1920,
        "renderExtParams": {
            "outputFilter": "depthPlane",
            "saveTiff": True,
            "uploadFolder": "muyu",
            "samplePerPixel": 1
        },
        "skybox": {
            "uvScaleX": 1,
            "localOffsetY": 0,
            "label": "",
            "type": "builtin",
            "fillMode": 1,
            "intensity": 3,
            "rotationY": 154,
            "bgColor": [
                233,
                233,
                233
            ],
            "mappingType": 0,
            "name": "sunny_vondelpark",
            "texelSize": 1024,
            "uvOffsetY": 0,
            "uvScaleY": 1,
            "uvOffsetX": 0,
            "uvScale": 1
        },
        "designId": "11f7d8db-94dd-495c-bf77-07daf9ef0bfe",
        "sceneUrl": "https://ossgw.alicdn.com/homeai-inner/material/barrgan_render_data/0f6d254f-765c-d0c4-490e-8c867254ea97_scene_1642128679876.json",
    }


def gen_dataset_request(src_file, request_file):
    batch_params = []
    params = json.load(open(src_file))
    for idx, param in enumerate(params):
        request_param = json.loads(param['requestParam'])
        render_param = request_param['renderParams']
        batch_param = deepcopy(MOCK)
        batch_param['camera'] = render_param['camera']
        task_id = f'scene-00-{idx:04d}'
        batch_param['bizId'] = task_id
        batch_param['imageName'] = task_id
        batch_params.append(batch_param)
    
    json.dump(batch_params, open(request_file, 'w'), indent=2)


def bspline(cv, n=100, degree=3):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate result
    bspl=BSpline(kv,cv, degree)
    return bspl(u)


def affine_from_data(camera, height, width):
    # height, width = data['param']['height'], data['param']['width']
    fovy = camera['fov']
    angle_y = fovy * np.pi / 180
    fl_y = height / np.tan(angle_y/2) / 2

    # forward
    target = np.array(camera['target'])
    pos = np.array(camera['pos'])
    forward = target-pos
    forward = forward / np.linalg.norm(target-pos)
    # right
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    # down
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    affine = np.vstack([right, down, forward, pos]).T
    affine = np.vstack([affine, np.array([0, 0, 0, 1])])
    return fl_y, affine


def parse_render_result(csv_file):
    tiff_prefix = 't3d-engine-lib/aceray_output/muyu/'
    output_dir = dirname(csv_file)
    depth_dir = pjoin(output_dir, 'depth')
    image_dir = pjoin(output_dir, 'images')
    if not exists(depth_dir): makedirs(depth_dir)
    if not exists(image_dir): makedirs(image_dir)
    oss = OssManager('t3d-publish', region='oss-cn-zhangjiakou')
    df = pd.read_csv(csv_file)
    transforms = {
        "coordinate": "muyu_synthetic",
        "depth_scaling_factor": 1.0,
        "scale": 1.0,
        "depth_scale": 1.0,
        "offset": [0.0, 0.0, 0.0],
        "frames": []
    }
    for rid, row in tqdm(df.iterrows(), total=len(df)):
        ext = json.loads(row['ext'])
        context = json.loads(ext['context'])
        data = context['bizData']
        task_id = data['param']['bizId']
        oss_key = data['RENDER_RESULT']['t3dRenderResult']['resultOssKey']
        tiff_name = splitext(basename(oss_key))[0]+'.tiff'
        tiff_name = pjoin(tiff_prefix, tiff_name)
        img_url = data['RENDER_RESULT']['t3dRenderResult']['resultUrl']

        depth = oss.get_object(tiff_name)
        open(pjoin(depth_dir, f'{task_id}.tiff'), 'wb').write(depth)
        img = requests.get(img_url, stream=True)
        img_ext = splitext(basename(img_url))[1]
        copyfileobj(img.raw, open(pjoin(image_dir, f'{task_id}{img_ext}'), 'wb'))

        height, width = data['param']['height'], data['param']['width']
        fl_y, affine = affine_from_data(data['param']['camera'], height, width)
        
        frame = {
            "file_path": pjoin('images', f'{task_id}{img_ext}'),
            "depth_file_path": pjoin('depth', f'{task_id}.tiff'),
            "fx": fl_y,
            "fy": fl_y,
            "cx": width / 2,
            "cy": height / 2,
            "transform_matrix": affine.tolist()
        }
        transforms['frames'].append(frame)
        pass
    
    json.dump(transforms,
              open(pjoin(output_dir, 'transforms_aligned.json'), 'w'),
              indent=2)
    pass


def parse_animation(animation_file):
    animation = json.load(open(animation_file))
    
    transforms = {
        "coordinate": "muyu_synthetic",
        "depth_scaling_factor": 1.0,
        "scale": 1.0,
        "depth_scale": 1.0,
        "offset": [0.0, 0.0, 0.0],
        "frames": []
    }

    H, W, n_step = 1080, 1920, 50
    cam_pts = []
    frames = []
    for seg in animation['effects']:
        n_seg = len(seg['status'])-1
        n_frames = n_step*n_seg
        pos_seg = np.array([d['pos'] for d in seg['status']])
        pos = bspline(pos_seg, n=n_frames, degree=2)
        cam_pts.append(pos)

        affines = [affine_from_data(c, H, W) for c in seg['status']]
        rots = np.stack([aff[1][:3,:3] for aff in affines], axis=0)
        rots = Rotation.from_matrix(rots)
        times = np.arange(len(rots))
        slp = Slerp(times, rots)
        times = np.linspace(times[0], times[-1], n_frames)
        slp_rots = slp(times)

        fl_y = affines[0][0]
        seg_frames = []
        for frame_idx in range(n_frames):
            affine = np.eye(4)
            affine[:3, 3] = pos[frame_idx]
            affine[:3, :3] = slp_rots[frame_idx].as_matrix()
            seg_frames.append({
                "file_path": '',
                "depth_file_path": '',
                "fx": fl_y,
                "fy": fl_y,
                "cx": W / 2,
                "cy": H / 2,
                "transform_matrix": affine.tolist()
            })
            pass
        frames += seg_frames
        pass
    cam_pts = np.vstack(cam_pts)
    mesh = Trimesh(vertices=cam_pts, vertex_colors=np.zeros_like(cam_pts), process=False)
    open(pjoin(dirname(animation_file), f'cam_pts.ply'), 'wb').write(export_ply(mesh))

    transforms['frames'] = frames
    json.dump(transforms,
              open(pjoin(dirname(animation_file), 'transforms_video.json'), 'w'),
              indent=2)
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='action')
    parser.add_argument('--req_src_file', type=str, help='req_src_file', default='')
    parser.add_argument('--req_out_file', type=str, help='req_out_file', default='')
    parser.add_argument('--render_csv', type=str, help='render_csv', default='')
    parser.add_argument('--animation_file', type=str, help='animation_file', default='')
    opt = parser.parse_args()

    if 'gen_request' == opt.action:
        gen_dataset_request(opt.req_src_file, opt.req_out_file)
    elif 'parse_render_result' == opt.action:
        parse_render_result(opt.render_csv)
    elif 'parse_animation' == opt.action:
        parse_animation(opt.animation_file)
    pass