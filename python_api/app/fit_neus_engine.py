import gin
import numpy as np
import torch
import tqdm
import shutil

from tensorboardX import SummaryWriter
from os import makedirs
from os.path import exists, join as pjoin, samefile
from PIL import Image
from python_api.app.engine import Engine
from python_api.app.config import Config

from python_api.metrics import Loss_factory
from python_api.provider import get_dataset, prepare_data, DataTransformer


def save_checkpoint(state_dict, save_path, device):
    if exists(save_path):
        last_state_dict = torch.load(save_path, map_location=device)
        if last_state_dict['best_metric'] > state_dict['best_metric']:
            return
    torch.save(state_dict, save_path)


def main(config_file):
    gin.parse_config_files_and_bindings([config_file], None)
    config = Config()
    if not exists(config.exp_dir): makedirs(config.exp_dir)
    try:
        shutil.copy(config_file, pjoin(config.exp_dir, 'config.gin'))
    except shutil.SameFileError:
        pass

    print('==> build NGP and renderer')
    engine = Engine(config)
    device = engine.device

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print('==> build dataset')
    dataset = get_dataset('train', config.data_dir, config)
    data_trans = DataTransformer(dataset, device, config)
    val_dataset = get_dataset('test', config.data_dir, config)

    print('==> init optimize routine')
    # criterion = torch.nn.HuberLoss(delta=0.1)
    criterion = Loss_factory.build(**config.criterion)
    # criterion = torch.nn.MSELoss()
    # criterion_depth = torch.nn.HuberLoss(delta=0.1)
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

    optim_params = []
    optim_params += engine.collect_parameters()
    optim_params += data_trans.collect_parameters()

    optimizer = torch.optim.Adam(optim_params,
                                 lr=config.lr_init,
                                 betas=(0.9, 0.99),
                                 eps=1e-15)

    summary_dir = pjoin(config.exp_dir, "run")
    if not exists(summary_dir): makedirs(summary_dir)
    writer = SummaryWriter(summary_dir)
    valid_dir = pjoin(config.exp_dir, "validation")
    if not exists(valid_dir): makedirs(valid_dir)

    best_ckpt = pjoin(config.exp_dir, 'ckpt_best.pth.tar')
    if exists(best_ckpt):
        state_dict = torch.load(best_ckpt, map_location=device)
        optimizer.load_state_dict(state_dict['optimizer'])
        step = state_dict['step']
        total_loss = state_dict['total_loss']
        config.lrate_decay = state_dict['lrate_decay']
        config.lr_init = state_dict['lr_init']
        engine.load_ngp(state_dict)
        data_trans.load_state(state_dict)
    else:
        total_loss = 0
        step = 1  # state.optimizer.state.step + 1

    pbar = tqdm.tqdm(total=config.max_steps, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    pbar.update(step-1)

    print('==> start fitting NGP')
    # for step, batch in zip(range(init_step, config.max_steps + 1), dataset):
    for epoch in range(config.max_epochs):
        if step >= config.max_steps + 1: break
        for batch in dataset:
            if step >= config.max_steps + 1: break

            batch = data_trans.preprocess_data(batch)
            rays, pixels_gt = batch['rays'], batch['pixels']
            optimizer.zero_grad()
            rets = engine.run(rays)

            ##### blender the foreground and background
            rets = rets[-2:]  # extract inside integration and outside
            weights_sum = rets[0].weights.sum(dim=-1, keepdim=True)
            fore_colors = rets[0].pixels.colors
            back_colors = rets[1].pixels.colors
            pixels = rets[1].pixels._replace(colors=fore_colors+back_colors*(1-weights_sum))
            rets[1]._replace(pixels=pixels)
            rets[0]._replace(pixels=None)  # only use the blender result for loss calculation


            rets = data_trans.postprocess_data(rets, batch)

            loss_dict = criterion(rets, batch)
            loss, loss_stat = engine.parse_loss_info(loss_dict)

            psnr = mse2psnr(img2mse(rets[-1].pixels.colors, pixels_gt))
            psnr = psnr.item()

            avg_acc = rets[-1].weights.sum(-1).mean().item()

            loss.backward()
            optimizer.step()

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
                for loss_k, loss_v in loss_stat.items():
                    writer.add_scalar(f"train/{loss_k}", loss_v, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
                writer.add_scalar("train/psnr", psnr, step)
                writer.add_scalar("train/acc", avg_acc, step)
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/step:.4f}), psnr={psnr:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
                pbar.update(config.print_every)

            if step % config.save_every == 0:
                test_batch = next(val_dataset)
                test_batch = prepare_data(test_batch, engine.device)
                test_rays, test_pixels_gt = test_batch['rays'], test_batch['pixels']

                test_pixels, depth, acc = engine.draw(test_rays, val_dataset.batch_size)
                test_pixels = data_trans.postprocess_data_eval(test_pixels, test_batch)
                acc = acc.cpu().numpy()
                depth = depth.cpu().numpy()
                acc_mask = acc > 0.9
                depth_w_mask = engine.visualize_depth(depth, acc_mask)
                depth = engine.visualize_depth(depth)
                acc = engine.visualize_depth(1-acc, depth_min=0, depth_max=1, direct=True)

                # test_loss = criterion(test_pixels, test_pixels_gt).item()
                test_psnr = mse2psnr(img2mse(test_pixels, test_pixels_gt))
                test_psnr = test_psnr.item()
                img = to8b(test_pixels.cpu().numpy())
                writer.add_image('test/rgb', img.transpose((2, 0, 1)), step)
                writer.add_image('test/depth', depth.transpose((2, 0, 1)), step)
                # writer.add_scalar("test/loss", test_loss, step)
                writer.add_scalar("test/psnr", test_psnr, step)

                Image.fromarray(img).save(pjoin(valid_dir, f'{step-1:04d}.png'))
                Image.fromarray(depth).save(pjoin(valid_dir, f'{step-1:04d}_depth.png'))
                Image.fromarray(depth_w_mask).save(pjoin(valid_dir, f'{step-1:04d}_depth_w_mask.png'))
                Image.fromarray(acc).save(pjoin(valid_dir, f'{step-1:04d}_acc.png'))

                state = {
                    'step': step,
                    'total_loss': total_loss,
                    'best_metric': test_psnr,
                    **engine.export_ngp(),
                    **data_trans.export_state(),
                    'lrate_decay': config.lrate_decay,
                    'lr_init': config.lr_init,
                    'optimizer': optimizer.state_dict()
                }
                save_checkpoint(state, best_ckpt, device)

            step += 1

    save_checkpoint(
        {
            'step': step,
            'total_loss': total_loss,
            'best_metric': test_psnr,
            **engine.export_ngp(),
            **data_trans.export_state(),
            'lrate_decay': config.lrate_decay,
            'lr_init': config.lr_init,
            'optimizer': optimizer.state_dict()
        },
        pjoin(config.exp_dir, f'ckpt_{step}.pth.tar'),
        device
    )

    writer.close()
    print('==> end fitting NGP')
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path')
    opt = parser.parse_args()
    main(opt.config_path)
