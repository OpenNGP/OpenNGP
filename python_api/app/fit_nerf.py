import gin
import dataclasses
import torch
import torch.optim as optim

from python_api.primitive.base_nerf import BaseNeRF
from python_api.provider.dataset import get_dataset
from python_api.renderer.renderer import Renderer
from python_api.utils.data_helper import namedtuple_map


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""
    data_dir: str = ''  # data_dir
    dataset_loader: str = 'multicam'  # The type of dataset loader to use.
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 4096  # The number of rays/pixels in each batch.
    factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    spherify: bool = False  # Set to True for spherical 360 scenes.
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    lr_init: float = 5e-4  # The initial learning rate.
    lr_final: float = 5e-6  # The final learning rate.
    lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
    lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
    max_steps: int = 1000000  # The number of optimization steps.
    save_every: int = 100000  # The number of steps to save a checkpoint.
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


def main():
    gin.parse_config_files_and_bindings(['test/test.gin'], None)
    config = Config()
    ngp = BaseNeRF()
    renderer = Renderer(config.render_passes)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ngp.to(device)

    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.HuberLoss(delta=0.1)

    optimizer = torch.optim.Adam([
        {'name': 'net', 'params': ngp.parameters(), 'weight_decay': 5e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    milestones = [2000, 4000, 6000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.33)

    init_step = 0  # state.optimizer.state.step + 1
    dataset = get_dataset('train', config.data_dir, config)
    for step, batch in zip(range(init_step, config.max_steps + 1), dataset):
        rays = namedtuple_map(
            lambda r: torch.from_numpy(r).to(device, non_blocking=True),
            batch['rays']
        )
        pixels_gt = torch.from_numpy(batch['pixels']).to(device, non_blocking=True)
        rets = renderer.render(rays, ngp)
        optimizer.zero_grad()
        loss = 0
        for ret in rets:
            loss += criterion(ret.pixels.colors, pixels_gt)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    pass


if __name__ == '__main__':
    main()
