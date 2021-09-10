#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
----------------------------
@ Author: ID:768           -
----------------------------
@ function:

@ Version:

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from scipy import linalg as la
import random

### A lightweight deep network ###
class CSP(torch.nn.Module):
    def __init__(self, upscale=4, mid_chn=64):
        super(CSP, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(1, mid_chn,  [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(mid_chn, mid_chn, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(mid_chn, mid_chn, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(mid_chn, mid_chn, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(mid_chn, mid_chn, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(mid_chn, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        xpad = F.pad(x_in, (0, 1, 0, 1), mode='reflect')
        # xpad = x_in
        B, C, H, W = xpad.shape
        xpad = xpad.reshape(B*C, 1, H, W)

        x = self.conv1(xpad)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        return x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks  = UNetConvBlock(in_size, out_size)
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.blocks(x)
        return x + sc

class SubspaceProjectRecModule(nn.Module):
    def __init__(self, src_ch, out_ch, num_subspace=8):
        super(SubspaceProjectRecModule, self).__init__()
        self.num_subspace = num_subspace
        self.subnet = Subspace(src_ch*2, num_subspace)
        self.conv_block = UNetConvBlock(src_ch*2, out_ch)

    def forward(self, x, bridge):
        b_, c_, h_, w_ = bridge.shape
        out = torch.cat([x, bridge], 1)
        sub = self.subnet(out)
        V_t = sub.reshape(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = torch.transpose(V_t, 1, 2)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.reshape(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, torch.transpose(bridge_, 1, 2))
        bridge = torch.matmul(V, project_feature)
        bridge = torch.transpose(bridge, 1, 2).reshape(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out


class NBInvBlockExp(nn.Module):
    def __init__(self, channel_num, channel_L, clamp=1., mid_channel=64):
        super(NBInvBlockExp, self).__init__()
        self.split_Hlen = channel_num - channel_L
        self.split_Llen = channel_L
        # self.split_Nlen = channel_num - channel_H - channel_L
        self.clamp = clamp
        self.recH = SubspaceProjectRecModule(src_ch=self.split_Hlen, out_ch=self.split_Hlen)

    def forward(self, hx, h):
        xL, xH = (hx.narrow(1, 0, self.split_Llen), hx.narrow(1, self.split_Llen, self.split_Hlen))

        xH = self.recH(xH, h)

        return torch.cat((xL, xH), dim=1)



class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, mid_chn=64):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, mid_chn, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(mid_chn, mid_chn, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(mid_chn, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num,
                 channel_split_num, clamp=1., mid_channel=64):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, mid_channel=mid_channel)
        self.G = subnet_constructor(self.split_len1, self.split_len2, mid_channel=mid_channel)
        self.H = subnet_constructor(self.split_len1, self.split_len2, mid_channel=mid_channel)

    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        y1 = x1 + self.F(x2)
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        return torch.cat((y1, y2), 1)

class NHL(nn.Module):
    def __init__(self, subnet_constructor, channel_num,
                 channel_H, channle_L, clamp=1., mid_channel=32):
        super(NHL, self).__init__()
        self.split_Hlen = channel_H
        self.split_Llen = channle_L
        self.split_Nlen = channel_num - channel_H - channle_L
        self.inv_NH = MSRes(subnet_constructor, channel_num=self.split_Nlen + self.split_Hlen,
                                  channel_split_num=self.split_Hlen, mid_channel=mid_channel//2)

        self.inv_HL = MSRes(subnet_constructor, channel_num=self.split_Llen + self.split_Hlen,
                                  channel_split_num=self.split_Llen, mid_channel=mid_channel//2)

    def forward(self, x):
        xL, xH, xN = (x.narrow(1, 0, self.split_Llen),
                      x.narrow(1, self.split_Llen, self.split_Hlen),
                      x.narrow(1, self.split_Hlen, self.split_Nlen))
        xHN = self.inv_NH(torch.cat([xH, xN], dim=1))
        xH, xN = (xHN.narrow(1, 0, self.split_Hlen), xHN.narrow(1, self.split_Hlen, self.split_Nlen))
        xLH = self.inv_HL(torch.cat([xL, xH], dim=1))
        xL = xLH.narrow(1, 0, self.split_Llen)
        return torch.cat((xL, xH, xN), 1)

class MSRes(nn.Module):
    def __init__(self, subnet_constructor, channel_num,
                 channel_split_num, clamp=1., mid_channel=32):
        super(MSRes, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len1, self.split_len1, mid_channel=mid_channel)
        self.G = subnet_constructor(self.split_len2, self.split_len2, mid_channel=mid_channel)
        self.H = subnet_constructor(channel_num, channel_num, mid_channel=mid_channel)

    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        y1 = x1 + self.F(x1)
        y2 = x2 + self.G(x2)
        y = torch.cat((y1, y2), 1)
        y = y + self.H(y)
        return y

class TSB(nn.Module):
    def __init__(self, subnet_constructor, current_channel, channel_out, cal_jacobian=False):
        super(TSB, self).__init__()
        self.cal_jacobian=cal_jacobian
        self.subInvBlk1 = NHL(subnet_constructor, current_channel, channle_L=channel_out, channel_H=channel_out)
        self.subInvBlk2 = NHL(subnet_constructor, current_channel, channle_L=channel_out, channel_H=channel_out)
        self.subInvBlk3 = MSRes(subnet_constructor, current_channel*2, current_channel, mid_channel=32)

    def forward(self, up, btm):
        up = self.subInvBlk1(up)
        up = self.subInvBlk2(up)
        x = torch.cat([up, btm], dim=1)
        x = self.subInvBlk3(x)
        up, btm = x.chunk(2, dim=1)
        return up, btm


class TSE(torch.nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(TSE, self).__init__()
        self.down_num = down_num
        self.block_num = block_num
        # operations = []
        self.blk_ops = nn.ModuleList()
        current_channel = channel_in
        # self.squeezeF = SqueezeFunction()
        self.haar_downsample = nn.ModuleList()
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            self.haar_downsample.append(b)
            current_channel *= 4
            operations = nn.ModuleList()
            for j in range(block_num[i]):
                b = TSB(subnet_constructor, current_channel, channel_out)
                operations.append(b)
            # operations = nn.Sequential(*operations)
            self.blk_ops.append(operations)
        self.noise_pred = Noise_Model_Network(channels=current_channel, filters_pack=current_channel - channel_out)

    def forward(self, x):
        up = x
        btm = x
        out_Hs = []
        for d_idx in range(self.down_num):
            up = self.haar_downsample[d_idx](up)
            btm = self.haar_downsample[d_idx](btm)
            for blk_op in self.blk_ops[d_idx]:
                up, btm = blk_op(up, btm)
            if d_idx == self.down_num-1:
                nleve = self.noise_pred(btm)
                lq = up[:, :3, :, :]
                hq = up[:, 3:, :, :]
                chq = hq * nleve + hq
                up = torch.cat([lq, chq], dim=1)
            out_Hs.append(up[:, 3:, :, :])
        return up, out_Hs


class SPR(nn.Module):
    def __init__(self, subnet_constructor=None):
        super(SPR, self).__init__()
        self.haardp1 = HaarDownsampling(3)
        # self.resblk1 = subnet_constructor(9, 9)
        self.lrankrec1 = NBInvBlockExp(12, 3)
        self.invblk1_1 = MSRes(subnet_constructor, 12,3)
        self.invblk1_2 = MSRes(subnet_constructor, 12,3)

        self.haardp2 = HaarDownsampling(12)
        # self.resblk2 = subnet_constructor(45, 45)
        self.lrankrec2 = NBInvBlockExp(48, 3)
        self.invblk2_1 = MSRes(subnet_constructor, 48,3)
        self.invblk2_2 = MSRes(subnet_constructor, 48,3)

        self.invblk3_1 = MSRes(subnet_constructor, 12, 3)
        self.invblk3_2 = MSRes(subnet_constructor, 12, 3)

    def forward(self, pseudoSR, x_Hs):
        x_H1, x_H2 =  x_Hs[0], x_Hs[1],

        hx = self.haardp1(pseudoSR)

        # in_L, in_H = hx[:, :3, :, :], hx[:, 3:, :, :]
        # in_H = self.resblk1(in_H) * x_H1 + in_H
        in_LH = self.lrankrec1(hx, x_H1)
        # in_LH = torch.cat((in_L, in_H), 1)
        out_LH = self.invblk1_1(in_LH)
        out_LH = self.invblk1_2(out_LH)

        hx = self.haardp2(out_LH)
        # in_L, in_H = hx[:, :3, :, :], hx[:, 3:, :, :]
        # in_H = self.resblk2(in_H) * x_H2 + in_H
        # in_LH = torch.cat((in_L, in_H), 1)
        in_LH = self.lrankrec2(hx, x_H2)
        out_LH = self.invblk2_1(in_LH)
        out_LH = self.invblk2_2(out_LH)

        hx = self.haardp2(out_LH, True)
        in_LH = self.invblk3_1(hx)
        in_LH = self.invblk3_2(in_LH)

        hx = self.haardp1(in_LH, True)

        hx = hx + pseudoSR
        return hx


class Noise_Model_Network(nn.Module):
    def __init__(self, channels=3, filters_num = 128, filters_pack = 4):
        super(Noise_Model_Network, self).__init__()

        # Noise Model Network

        self.conv_1 = nn.Conv2d(channels, filters_num, 1, 1, 0, groups=1)

        self.conv_2 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_3 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_4 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_5 = nn.Conv2d(filters_num, filters_pack, 1, 1, 0, groups=1)
        self.rlu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.rlu(self.conv_1(x))
        x = self.rlu(self.conv_2(x))
        x = self.rlu(self.conv_3(x))
        x = self.rlu(self.conv_4(x))
        x = self.rlu(self.conv_5(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class Network(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, subnet_constructor=None, block_num=[], down_num=2):
        super(Network, self).__init__()
        self.tse = TSE(channel_in=in_ch, channel_out=out_ch,
                                             subnet_constructor=subnet_constructor,
                                             block_num=block_num, down_num=down_num)
        self.srDecoder = CSP(upscale=down_num*2, mid_chn=32)
        self.refiner = SPR(subnet_constructor=subnet_constructor)

    def forward(self, x):
        ori = x
        x, x_Hs = self.tse(x)
        x_L = x[:, :3, :, :]
        sr_L1 = self.srDecoder(x_L)
        sr_L2 = self.refiner(sr_L1, x_Hs)

        return sr_L2, sr_L1, x_L


def define_G(opt):
    from models.modules.Subnet_constructor import subnet
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = Network(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), [4, 4], down_num)

    return netG


if __name__ == '__main__':
    import argparse
    import options.options as option

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    model = define_G(opt)
    # x = torch.rand(1, 3, 128, 128)
    # o = model(x)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)

    # 256
    # Flops: 35.68 GMac
    # Params: 3.74 M