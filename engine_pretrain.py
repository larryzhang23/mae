# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import numpy as np
import torch
import wandb
from mmcv.image import tensor2imgs
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if args.wandb and misc.is_main_process() and data_iter_step % print_freq == 0:
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', lr, epoch_1000x)
            wandb.log({'loss': loss_value_reduce, 'lr': lr})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def patches_mask_to_img_mask(mask, patch_size=16):
    # mask: B, L
    B, L = mask.size()
    h = w = int(L ** 0.5)
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size **2 * 3)
    mask = mask.reshape(B, h, w, patch_size, patch_size, 3)
    # mask: B, C, H, W
    mask = torch.einsum('bhwpqc->bhpwqc', mask)
    return mask.reshape(B, h * patch_size, w * patch_size, 3)


def visualize(model, dataloader, args):
    vis_lst = []
    with torch.no_grad():
        for i, (samples, _) in enumerate(dataloader):
            if i >= 1:
                break
            samples = samples.to(next(model.parameters()).device, non_blocking=True)

            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
            if hasattr(model, 'module'):
                model = model.module
            recons = model.unpatchify(pred)
            mask = patches_mask_to_img_mask(mask)
            for i, img, per_mask, recon in zip(range(16), samples, mask, recons):
                recon = tensor2imgs(recon.unsqueeze(0), mean=[123.675, 116.28, 103.53],
                                     std=[58.395, 57.12, 57.375],
                                     to_rgb=False)[0]
                img = tensor2imgs(img.unsqueeze(0), mean=[123.675, 116.28, 103.53],
                                     std=[58.395, 57.12, 57.375],
                                     to_rgb=False)[0]
                per_mask = per_mask.cpu().numpy()
                recon = recon * per_mask + img * (1 - per_mask)
                mask_img = img * (1 - per_mask)
                concat = np.concatenate([img, mask_img, recon], axis=1)
                vis_lst.append(wandb.Image(concat))
    return vis_lst