import gin
import numpy as np
import torch
import tqdm

from tensorboardX import SummaryWriter
from os import makedirs
from os.path import exists, join as pjoin
from python_api.app.engine import Engine
from python_api.app.config import Config

from python_api.metrics import Loss_factory
from python_api.provider import get_dataset
from python_api.provider.dataset_ngp import NeRFRayDataset
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

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print('==> build dataset')
    if config.torch_dataset:
        dataset = NeRFRayDataset('train', config.data_dir, config)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    else:
        dataset = get_dataset('train', config.data_dir, config)
    val_dataset = get_dataset('test', config.data_dir, config)

    print('==> init optimize routine')
    # criterion = torch.nn.HuberLoss(delta=0.1)
    criterion = Loss_factory.build(**config.criterion)
    # criterion = torch.nn.MSELoss()
    # criterion_depth = torch.nn.HuberLoss(delta=0.1)
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

    optimizer = torch.optim.Adam([
        {'name': 'net', 'params': engine.collect_parameters()},
    ], lr=config.lr_init, betas=(0.9, 0.99), eps=1e-15)

    summary_dir = pjoin(config.exp_dir, "run")
    if not exists(summary_dir): makedirs(summary_dir)
    writer = SummaryWriter(summary_dir)

    best_ckpt = pjoin(config.exp_dir, 'ckpt_best.pth.tar')
    if exists(best_ckpt):
        state_dict = torch.load(best_ckpt, map_location=device)
        optimizer.load_state_dict(state_dict['optimizer'])
        step = state_dict['step']
        total_loss = state_dict['total_loss']
        config.lrate_decay = state_dict['lrate_decay']
        config.lr_init = state_dict['lr_init']
        engine.load_ngp(state_dict)
    else:
        total_loss = 0
        step = 1  # state.optimizer.state.step + 1

    pbar = tqdm.tqdm(total=config.max_steps, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    pbar.update(step-1)

    print('==> start fitting NGP')
    # for step, batch in zip(range(init_step, config.max_steps + 1), dataset):
    for batch in dataset:
        if step >= config.max_steps + 1: break
        batch = engine.prepare_data(batch)
        rays, pixels_gt = batch['rays'], batch['pixels']
        optimizer.zero_grad()
        rets = engine.run(rays)

        loss_dict = criterion(rets, batch)
        loss, loss_stat = engine.parse_loss_info(loss_dict)

        # # color
        # for ret in rets:
        #     loss += criterion(ret.pixels.colors, pixels_gt)
        # # depth
        # for ret in rets:
        #     pred_depth = ret.pixels.depths[..., None][rays.mask]
        #     gt_depth = rays.depth[rays.mask]
        #     loss += criterion_depth(pred_depth, gt_depth) / config.bound

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
            test_batch = engine.prepare_data(test_batch)
            test_rays, test_pixels_gt = test_batch['rays'], test_batch['pixels']

            chunk = val_dataset.batch_size
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
                # test_loss = criterion(test_pixels, test_pixels_gt).item()
                test_psnr = mse2psnr(img2mse(test_pixels, test_pixels_gt))
                test_psnr = test_psnr.item()
                img = to8b(test_pixels.cpu().numpy()).transpose((2, 0, 1))
            writer.add_image('test/rgb', img, step)
            # writer.add_scalar("test/loss", test_loss, step)
            writer.add_scalar("test/psnr", test_psnr, step)

            state = {
                'step': step,
                'total_loss': total_loss,
                'best_metric': test_psnr,
                **engine.export_ngp(),
                'lrate_decay': config.lrate_decay,
                'lr_init': config.lr_init,
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(state, best_ckpt, device)

        step += 1

    writer.close()
    print('==> end fitting NGP')
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)
