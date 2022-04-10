import gin
import dataclasses
import numpy as np
import torch
import torch.optim as optim
import tqdm

from tensorboardX import SummaryWriter
from os import makedirs
from os.path import exists, join as pjoin
from rich.console import Console

from python_api.primitive.base_nerf import BaseNeRF
from python_api.provider.dataset import get_dataset
from python_api.renderer.renderer import Renderer
from python_api.utils.data_helper import namedtuple_map


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""
    data_dir: str = ''  # data_dir
    exp_dir: str = ''  # exp_dar
    dataset_loader: str = 'multicam'  # The type of dataset loader to use.
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 4096  # The number of rays/pixels in each batch.
    factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    spherify: bool = False  # Set to True for spherical 360 scenes.
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    lr_init: float = 5e-4  # The initial learning rate.
    lrate_decay: int = 500
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
    max_steps: int = 200000  # The number of optimization steps.
    save_every: int = 10000  # The number of steps to save a checkpoint.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    gc_every: int = 10000  # The number of steps between garbage collections.
    test_render_interval: int = 1  # The interval between images saved to disk.
    disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
    randomized: bool = True  # Use randomized stratified sampling.
    near: float = 2.  # Near plane distance.
    far: float = 6.  # Far plane distance.
    coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
    weight_decay_mult: float = 0.  # The multiplier on weight decay.
    white_bkgd: bool = True  # If True, use white as the background (black o.w.).
    render_passes: list = dataclasses.field(default_factory=list)
    precrop_iters: int = 0
    precrop_frac: float = 0.5


def main(config_file):
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()
    console = Console()

    console.print('==> build NGP and renderer')
    ngp_coarse = BaseNeRF()
    renderer_coarse = Renderer([config.render_passes[0]])

    ngp_fine = BaseNeRF()
    renderer_fine = Renderer([config.render_passes[1]])

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ngp_coarse.to(device)
    ngp_fine.to(device)

    console.print('NGP coarse Geometry: ', ngp_coarse.geometry)
    console.print('NGP coarse Appearance: ', ngp_coarse.appearance)

    console.print('NGP fine Geometry: ', ngp_fine.geometry)
    console.print('NGP fine Appearance: ', ngp_fine.appearance)

    console.print('==> build dataset')
    dataset = get_dataset('train', config.data_dir, config)
    val_dataset = get_dataset('test', config.data_dir, config)

    console.print('==> init optimize routine')
    # criterion = torch.nn.HuberLoss(delta=0.1)
    criterion = torch.nn.MSELoss()
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

    optimizer = torch.optim.Adam([
        {'name': 'net', 'params': ngp_coarse.parameters()+ngp_fine.parameters()},
    ], lr=config.lr_init, betas=(0.9, 0.99), eps=1e-15)

    milestones = [40000, 80000, 120000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.33)
    summary_dir = pjoin(config.exp_dir, "run")
    if not exists(summary_dir): makedirs(summary_dir)
    writer = SummaryWriter(summary_dir)

    total_loss = 0
    init_step = 1  # state.optimizer.state.step + 1
    pbar = tqdm.tqdm(total=config.max_steps, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    pbar.update(init_step-1)

    console.print('==> start fitting NGP')
    for step, batch in zip(range(init_step, config.max_steps + 1), dataset):
        rays = namedtuple_map(
            lambda r: torch.from_numpy(r).to(device, non_blocking=True),
            batch['rays']
        )
        pixels_gt = torch.from_numpy(batch['pixels']).to(device, non_blocking=True)
        optimizer.zero_grad()
        ret_coarse = renderer_coarse.render(rays, ngp_coarse)
        ret_fine = renderer_fine.render(rays, ngp_fine, ret_coarse[-1]._asdict())
        rets = ret_coarse + ret_fine
        loss = 0
        for ret in rets:
            loss += criterion(ret.pixels.colors, pixels_gt)

        psnr = mse2psnr(img2mse(rets[-1].pixels.colors, pixels_gt))
        psnr = psnr.item()

        loss.backward()
        optimizer.step()

        # scheduler.step()
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = config.lrate_decay * 1000
        new_lrate = config.lr_init * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        loss_val = loss.item()
        total_loss += loss_val

        if step % config.print_every == 0:
            writer.add_scalar("train/loss", loss_val, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
            writer.add_scalar("train/psnr", psnr, step)
            pbar.set_description(f"loss={loss_val:.4f} ({total_loss/step:.4f}), psnr={psnr:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
            pbar.update(config.print_every)

        if step % config.save_every == 0:
            test_batch = next(val_dataset)
            chunk = val_dataset.batch_size
            test_rays = test_batch['rays']
            height, width = test_rays[0].shape[:2]
            num_rays = height * width
            test_rays = namedtuple_map(
                lambda r: r.reshape((num_rays, -1)),
                test_rays
            )
            test_rays = namedtuple_map(
                lambda r: torch.from_numpy(r).to(device, non_blocking=True),
                test_rays
            )
            test_pixels_gt = torch.from_numpy(test_batch['pixels'])
            test_pixels_gt = test_pixels_gt.to(device, non_blocking=True)
            results = []
            with torch.no_grad():
                for i in range(0, num_rays, chunk):
                    # pylint: disable=cell-var-from-loop
                    chunk_rays = namedtuple_map(
                        lambda r: r[i:i + chunk],
                        test_rays
                    )
                    test_ctx = {'perturb': False}
                    ret_coarse = renderer_coarse.render(chunk_rays, ngp_coarse, test_ctx)
                    test_ctx.update(ret_coarse[-1]._asdict())
                    ret_fine = renderer_fine.render(chunk_rays, ngp_fine, test_ctx)
                    results.append(ret_fine[-1].pixels.colors)
                test_pixels = torch.concat(results)
                test_pixels = test_pixels.reshape((height, width, -1))
                test_loss = criterion(test_pixels, test_pixels_gt).item()
                test_psnr = mse2psnr(img2mse(test_pixels, test_pixels_gt))
                test_psnr = test_psnr.item()
                img = to8b(test_pixels.cpu().numpy()).transpose((2, 0, 1))
            writer.add_image('test/rgb', img, step)
            writer.add_scalar("test/loss", test_loss, step)
            writer.add_scalar("test/psnr", test_psnr, step)

    writer.close()
    console.print('==> end fitting NGP')
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)
