import gin
import dataclasses


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""
    data_dir: str = ''  # data_dir
    exp_dir: str = ''  # exp_dar
    eval_dir: str = 'eval'
    dataset_loader: str = 'multicam'  # The type of dataset loader to use.
    torch_dataset: bool = False
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
    primitives: list = dataclasses.field(default_factory=list)
    pipelines: list = dataclasses.field(default_factory=list)
    eval_pipelines: list = dataclasses.field(default_factory=list)
    criterion: dict = dataclasses.field(default_factory=dict)
    precrop_iters: int = 0  # crop center for early training
    precrop_frac: float = 0.5
    lazy_ray: bool = False  # lazy generate ray to save memory
    load_depth: bool = False
    bound: float = 1.0
