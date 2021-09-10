#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
----------------------------
@ Author: ID 768           -
----------------------------
@ function:

@ Version:

"""
import time
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from tqdm import tqdm
import cv2

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            pass
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    model = create_model(opt)
    model.load_network(opt['path']['resume_state'], model.netG)

    avg_psnr = 0.0
    avg_time = 0.0
    idx = 0

    for val_data in tqdm(val_loader):
        idx += 1
        model.feed_data(val_data)
        t = time.time()
        model.test()
        intim = time.time() - t
        denoised_img = util.tensor2img(model.fake_RH)  # uint8
        gt_img = util.tensor2img(model.real_H)  # uint8

        crop_size = opt['scale']
        gt_img = gt_img / 255.
        denoised_img_psnr = denoised_img / 255.
        cropped_denoised_img = denoised_img_psnr[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        psnr = util.calculate_psnr(cropped_denoised_img * 255, cropped_gt_img * 255)
        print(psnr)
        avg_psnr += psnr
        avg_time += intim

    avg_psnr = avg_psnr / idx
    avg_time = avg_time / idx
    # log
    print('# Validation # PSNR: {:.4e}.'.format(avg_psnr))
    print('val')  # validation logger
    print('psnr: {:.4e}.'.format(avg_psnr))
    print('avg_time: {:.4e}.'.format(avg_time))
    # tensorboard logger


if __name__ == '__main__':
    main()
