# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate

from util.mydataset import MyDataset
from util import transform
from util.losses import DiceLoss

# import open_clip
# from prompt_ensemble import AnomalyCLIP_PromptLearner


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=336, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default=r'/home/data1/zhangzr22/LLaVA_DATA/mae_raw/mae_pretrain_vit_base.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_nonoise',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_nonoise',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    

    # 新添加的参数
    parser.add_argument('--dataDir', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir1', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/Casiav1', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir2', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/Columbia', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir3', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/autosplice', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir4', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/DSO-1', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir5', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/Korus_new', type=str,
                        help='choose the dir of dataset')
    parser.add_argument('--testdataDir6', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/NIST_pixel', type=str,
                    help='choose the dir of dataset')
    parser.add_argument('--testdataDir7', default=r'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/IMD2020', type=str,
                    help='choose the dir of dataset')
    parser.add_argument('--train_txtdir', default='train_fantastic.txt', type=str)
    parser.add_argument('--val_txtdir', default='test.txt', type=str)
    parser.add_argument('--threshold', default=0.5, type=float)

    return parser


def test(args):
    device = torch.device(args.device)
    
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size,
    )
    args.output_dir = '/home/data1/zhangzr22/LLaVA_DATA/mae_raw/output_dir_336_zuobi_v2_dice'
    checkpoint = torch.load(os.path.join(args.output_dir,'checkpoint-bestv1.pth'), map_location='cpu')
    # /home/data1/zhangzr22/LLaVA_DATA/mae_raw/output_dir_336_zuobi_v2_dice/checkpoint-best.pth
    
    print("Load well-trained checkpoint from: %s" % os.path.join(args.output_dir,'checkpoint-best.pth'))
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=True)
    model.to(device)
    # --------------------------------------------------------------------------------------------------------------
    dataset_val1 = MyDataset(root_dir=args.testdataDir1, names_file=os.path.join(args.testdataDir1, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val1 = torch.utils.data.SequentialSampler(dataset_val1)

    data_loader_val1 = torch.utils.data.DataLoader(
        dataset_val1, sampler=sampler_val1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # --------------------------------------------------------------------------------------------------------------
    dataset_val2 = MyDataset(root_dir=args.testdataDir2, names_file=os.path.join(args.testdataDir2, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val2 = torch.utils.data.SequentialSampler(dataset_val2)

    data_loader_val2 = torch.utils.data.DataLoader(
        dataset_val2, sampler=sampler_val2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # --------------------------------------------------------------------------------------------------------------
    dataset_val3 = MyDataset(root_dir=args.testdataDir3, names_file=os.path.join(args.testdataDir3, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val3 = torch.utils.data.SequentialSampler(dataset_val3)

    data_loader_val3 = torch.utils.data.DataLoader(
        dataset_val3, sampler=sampler_val3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # --------------------------------------------------------------------------------------------------------------
    dataset_val4 = MyDataset(root_dir=args.testdataDir4, names_file=os.path.join(args.testdataDir4, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val4 = torch.utils.data.SequentialSampler(dataset_val4)

    data_loader_val4 = torch.utils.data.DataLoader(
        dataset_val4, sampler=sampler_val4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # --------------------------------------------------------------------------------------------------------------
    dataset_val5 = MyDataset(root_dir=args.testdataDir5, names_file=os.path.join(args.testdataDir5, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val5 = torch.utils.data.SequentialSampler(dataset_val5)

    data_loader_val5 = torch.utils.data.DataLoader(
        dataset_val5, sampler=sampler_val5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    dataset_val6 = MyDataset(root_dir=args.testdataDir6, names_file=os.path.join(args.testdataDir6, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val6 = torch.utils.data.SequentialSampler(dataset_val6)

    data_loader_val6 = torch.utils.data.DataLoader(
        dataset_val6, sampler=sampler_val6,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    dataset_val7 = MyDataset(root_dir=args.testdataDir7, names_file=os.path.join(args.testdataDir7, args.val_txtdir),
                            crop_size=args.input_size, crop=False, transform=None)

    sampler_val7 = torch.utils.data.SequentialSampler(dataset_val7)

    data_loader_val7 = torch.utils.data.DataLoader(
        dataset_val7, sampler=sampler_val7,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    
    print('CASIAv1----------------------------------')
    test_stats1 = evaluate(data_loader_val1, model, device, args=args)
    print('Columbia---------------------------------')
    test_stats2 = evaluate(data_loader_val2, model, device, args=args)
    print('autosplice')
    test_stats3 = evaluate(data_loader_val3, model, device, args=args)
    print('dso---------------------------------')
    test_stats4 = evaluate(data_loader_val4, model, device, args=args)
    print('korus---------------------------------')
    test_stats5 = evaluate(data_loader_val5, model, device, args=args)
    print('NIST---------------------------------')
    test_stats6 = evaluate(data_loader_val6, model, device, args=args)
    print('IMD---------------------------------')
    test_stats7 = evaluate(data_loader_val7, model, device, args=args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)

    # 三个数据集一起测
    args.batch_size = 128
    test(args)
    ## python test.py --output_dir /home/data1/zhangzr22/LLaVA_DATA/mae_raw/output_dir_512_fantastic
