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
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

from util.losses import DiceLoss
from sklearn import metrics
from util.utils import calculate_pixel_f1
import torch.nn as nn


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, edges) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        ##################  text prompt  ##############################
        # prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
        # text_features = clipmodel.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        # text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
        # text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            outputs, loss = model(samples, targets)
            # loss = criterion(outputs, targets)

            pix_f1_train = []
            pix_auc_train = []
            for i in range(len(samples)):
                pd = outputs[i][0].detach().cpu().flatten().numpy()
                pd_bin = (pd > args.threshold).astype(np.float64)
                gt = targets[i][0].detach().cpu().flatten().numpy()
                gt = (gt > 0).astype(np.int32)
                pix_f1, _, _ = calculate_pixel_f1(pd_bin, gt)
                pix_f1_train.append(pix_f1)
                try:
                    pix_auc = metrics.roc_auc_score(gt, pd)
                except ValueError:
                    pix_auc = 0
                pix_auc_train.append(pix_auc)

            metric_logger.update(pix_f1=np.mean(pix_f1_train))
            metric_logger.update(pix_auc=np.mean(pix_auc_train))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = DiceLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 1000, header):
        images = batch[0]
        target = batch[1]
        edge = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        edge = edge.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, loss = model(images, target)
            # loss = criterion(output, target)

            pix_f1_test = []
            pix_auc_test = []
            for i in range(len(images)):
                pd = output[i][0].detach().cpu().flatten().numpy()
                pd_bin = (pd > args.threshold).astype(np.float64)
                gt = target[i][0].detach().cpu().flatten().numpy()
                gt = (gt > 0).astype(np.int32)
                pix_f1, _, _ = calculate_pixel_f1(pd_bin, gt)
                pix_f1_test.append(pix_f1)
                try:
                    pix_auc = metrics.roc_auc_score(gt, pd)
                except ValueError:
                    pass
                pix_auc_test.append(pix_auc)

            metric_logger.update(pix_f1=np.mean(pix_f1_test))
            metric_logger.update(pix_auc=np.mean(pix_auc_test))


        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}