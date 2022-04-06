from python_api.primitive.base_nerf import BaseNeRF
from python_api.renderer.renderer import Renderer


def main():
    ngp = BaseNeRF()
    renderer = Renderer()

    for img, intrinsic, pose in zip([], [], []):
        rays_o, rays_d = [], []  # get ray from pose and intrinsic
        colors = renderer.render_color(rays_o, rays_d, ngp)
        loss = (img-colors)**2
    pass


if __name__ == '__main__':
    main()
