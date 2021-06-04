#!/usr/bin/env python
# coding: utf-8

# In[221]:
import os.path
import pickle

import numpy as np
import PIL.Image as Image
from torch import nn

import torch
from torch.optim import LBFGS
import torch.nn.functional as F
import argparse
import random
import tqdm
import sys
import torchvision as tv
sys.path.append('./diffjpeg/')
import DiffJPEG
import sys
import math
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--bits', type=int, default=3)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--lr1', type=float, default=0.1)
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--steps1', type=int, default=2000)
parser.add_argument('--steps2', type=int, default=1000)
parser.add_argument('--max_iter', type=int, default=20)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--image_range', type=int, default=100)
parser.add_argument('--loss2', type=str, default='BCE_bit')
parser.add_argument('--save_path', type=str, default='result/')
parser.add_argument('--new_weight_loss', action='store_true')
parser.add_argument('--last_norm', action='store_true')
parser.add_argument('--do_jpg', action='store_true')
parser.add_argument('--small_size', action='store_true')
parser.add_argument('--smaller_size', action='store_true')
parser.add_argument('--soft_delta', action='store_true')
parser.add_argument('--quant', action='store_true')
parser.add_argument('--quant_last', action='store_true')
parser.add_argument('--quality', type=int, default=80)
parser.add_argument('--model', type=str, default='model0')
parser.add_argument('--down', type=int, default=1)
parser.add_argument('--flip', action='store_true')
parser.add_argument('--shuffle_layer', action='store_true')
parser.add_argument('--shuffle_layer1', action='store_true')
parser.add_argument('--shuffle_layer2', action='store_true')
parser.add_argument('--shuffle_layer3', action='store_true')
parser.add_argument('--shuffle_layer4', action='store_true')
parser.add_argument('--shuffle_layer5', action='store_true')
parser.add_argument('--shuffle_layer6', action='store_true')
parser.add_argument('--small_first_layer', action='store_true')
parser.add_argument('--small_first_layer1', action='store_true')
parser.add_argument('--small_first_layer2', action='store_true')
parser.add_argument('--match_norm', action='store_true')
parser.add_argument('--match_norm1', action='store_true')
parser.add_argument('--activate', type=str, default='leaky')
parser.add_argument('--number', type=int, default=5)
parser.add_argument('--start_idx', type=int, default=1)
parser.add_argument('--not_early', action='store_true')
parser.add_argument('--image_path', type=str, default=None)

args = parser.parse_args()


def shuffle_params(m):
    if type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))

class noneactivate(nn.Module):
    def __init__(self):
        super(noneactivate, self).__init__()
    def forward(self, m):
        return m

def activations():
    if args.activate == 'leaky':
        return nn.LeakyReLU(inplace=True)
    elif args.activate == 'relu':
        return nn.ReLU(inplace=True)
    elif args.activate == 'none':
        return noneactivate()
    elif args.activate == 'tanh':
        return nn.Tanh()
    elif args.activate == 'sigmoid':
        return nn.Sigmoid()



def shuffle_params_kaiming(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


def shuffle_params_xv(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight.data)
        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


#


def acc_bits(seq, target):
    seq = seq.view(-1, 7)
    target = target.view(-1, 7)
    result = seq == target
    result = result.float().sum(-1).int()
    return torch.bincount(result)


def acc_bits1(seq, target):
    seq = seq.view(-1, 7)
    target = target.view(-1, 7)
    result = seq != target
    result = result.float().sum(-1).int()
    return result


def check_no_error(seq, target):
    result = acc_bits(seq, target)
    # print(result)
    # print(result[:-2].sum())
    return result[:-2].sum() == 0


def bit_loss(seq, target):
    seq = torch.sigmoid(seq)
    seq = seq.view(-1, 7).float()
    target = target.view(-1, 7).float()
    result = seq * target + (1 - seq) * (1 - target)
    result = 1 - result[:, 0] * result[:, 1] * result[:, 2] * result[:, 3] * result[:, 4] * result[:, 5] * result[:, 6]
    return result.sum()


def new_loss(seq, target, loss):
    correct = acc_bits1((seq > 0).float(), target)
    index = torch.arange(1, 8)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = index <= correct[:, None]
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(2, labels)
    labels[:, 0] = 0
    labels[~mask] = 0
    labels = labels.detach()
    loss = torch.sort(loss.view(-1, 7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)

def new_loss3(seq, target, loss):
    correct = acc_bits1((seq > 0).float(), target)
    index = torch.arange(1, 8)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = index <= correct[:, None]
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(2, labels)
    labels[:, 0] = 0
    labels[~mask] = 0
    labels = (labels/labels.sum()) * labels.numel()
    labels = labels.detach()
    loss = torch.sort(loss.view(-1, 7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)

def new_loss4(seq, target, loss):
    correct = acc_bits1((seq > 0).float(), target)
    index = torch.arange(1, 8)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = index <= correct[:, None]
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(4, labels)
    labels[:, 0] = 0
    labels[~mask] = 0
    labels = (labels/labels.sum()) * labels.numel()
    labels = labels.detach()
    loss = torch.sort(loss.view(-1, 7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)

def new_loss5(seq, target, loss):
    correct = acc_bits1((seq > 0).float(), target)
    index = torch.arange(1, 8)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = index <= correct[:, None]
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(8, labels)
    labels[:, 0] = 0
    labels[~mask] = 0
    labels = (labels/labels.sum()) * labels.numel()
    labels = labels.detach()
    loss = torch.sort(loss.view(-1, 7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)

def new_loss_vk(seq, target, loss, C=4):
    correct = acc_bits1((seq>0).float(),target)
    index = torch.arange(7, 0, -1)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = correct[:, None] >= index
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(C, labels)
    labels[mask] = 1
    labels[:, 0] = 0
    #normalize
    labels = (labels/labels.sum()) * labels.numel()
    labels = labels.detach()
    num_zeros = (loss==0).sum().item()
    loss[loss==0] += (torch.rand(num_zeros)*1e-10).to('cuda')
    loss = torch.sort(loss.view(-1,7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)

def new_loss1(seq, target, loss):
    correct = acc_bits1((seq > 0).float(), target)
    index = torch.arange(1, 8)[None, :].repeat(correct.shape[0], 1).contiguous().to(target.device).int()
    mask = index <= correct[:, None]
    labels = torch.arange(7)[None, :].repeat(correct.shape[0], 1).float().to(target.device)
    labels = torch.pow(2, labels)
    # labels[:, 0] = 0
    labels[~mask] = 0
    labels = labels.detach()
    loss = torch.sort(loss.view(-1, 7), dim=-1, descending=True)
    return torch.sum(loss[0] * labels)


class normLayer(nn.Module):
    def __init__(self):
        super(normLayer, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        assert b == 1
        mean = x.view(c, -1).mean(-1)
        std = x.view(c, -1).std(-1)
        x = x - mean.reshape([1, c, 1, 1])
        x = x / (std + 1e-7).reshape([1, c, 1, 1])
        return x

class BasicDecoder1(nn.Module):


    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoder1, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def _conv2d(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size, stride=1),
                  nn.LeakyReLU(inplace=True),
                  normLayer()]
        layers += [
            self._conv2d(self.hidden_size, self.hidden_size, stride=2),
            nn.LeakyReLU(inplace=True),
            normLayer()]
        layers += [
            self._conv2d(self.hidden_size, self.hidden_size, stride=1),
            nn.LeakyReLU(inplace=True),
            normLayer()]
        layers += [self._conv2d(self.hidden_size, self.data_depth * 16, stride=2)]
        if self.last_norm:
            layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]


    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

class BasicDecoder2(nn.Module):

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  # nn.LeakyReLU(inplace=True),
                  normLayer()]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                # nn.LeakyReLU(inplace=True),
                normLayer()]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        if self.last_norm:
            layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoder2, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

class BasicDecoder3(nn.Module):


    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  nn.LeakyReLU(inplace=True),
                  ]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                ]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        # if self.last_norm:
        #     layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoder3, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

class BasicDecoderAc(nn.Module):


    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  activations(),
                  normLayer()]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                activations(),
                normLayer()]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        if self.last_norm:
            layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoderAc, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x


class BasicDecoder(nn.Module):


    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  nn.LeakyReLU(inplace=True),
                  normLayer()]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                normLayer()]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        if self.last_norm:
            layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

    def forward1(self, x):
        outs = []
        # x = torch.flip(x, [1])
        net = list(self._models[0].children())
        x = net[2](net[1](net[0](x)))
        outs.append(x.cpu().data)
        # x = torch.flip(x, [1])
        for ii in range(1, self.num_layers):
            x = net[2+ii*3](net[1+ii*3](net[0+ii*3](x)))
            outs.append(x.cpu().data)
            # x = torch.flip(x, [1])
        x = net[-1](x)
        outs.append(x.cpu().data)
        return outs


class BasicDecoderBN(nn.Module):

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  nn.LeakyReLU(inplace=True),
                  nn.BatchNorm2d(self.hidden_size)]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size)]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        if self.last_norm:
            layers += [nn.BatchNorm2d(self.hidden_size)]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoderBN, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm
        self.flip = args.flip

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        if self.flip:
            x = torch.flip(x, [1])
        net = list(self._models[0].children())
        x = net[2](net[1](net[0](x)))
        if self.flip:
            x = torch.flip(x, [1])
        for ii in range(1, self.num_layers):
            x = net[2+ii*3](net[1+ii*3](net[0+ii*3](x)))
            if self.flip:
                x = torch.flip(x, [1])
        x = net[-1](x)
        return x

    def forward1(self, x):
        outs = []
        x = torch.flip(x, [1])
        net = list(self._models[0].children())
        x = net[2](net[1](net[0](x)))
        outs.append(x.cpu().data)
        x = torch.flip(x, [1])
        for ii in range(1, self.num_layers):
            x = net[2+ii*3](net[1+ii*3](net[0+ii*3](x)))
            outs.append(x.cpu().data)
            x = torch.flip(x, [1])
        x = net[-1](x)
        outs.append(x.cpu().data)
        return outs

class BasicDecoderNorm(nn.Module):

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size//2)
        )

    def _build_models(self):
        layers = [self._conv2d(3, self.hidden_size),
                  nn.LeakyReLU(inplace=True),
                  normLayer()]
        for _ in range(self.num_layers - 1):
            layers += [
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                normLayer()]
        layers += [self._conv2d(self.hidden_size, self.data_depth)]
        if self.last_norm:
            layers += [normLayer()]
        self.layers = nn.Sequential(*layers)
        return [self.layers]

    def __init__(self, data_depth, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoderNorm, self).__init__()
        self.kernel_size = kernel_size
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_norm = last_norm

        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = [self.layers]

            self.version = '1'

    def forward(self, x):
        x = torch.flip(x, [1])
        net = list(self._models[0].children())
        x = net[2](net[1](net[0](x)))
        x = torch.flip(x, [1])
        for ii in range(1, self.num_layers):
            x = net[2+ii*3](net[1+ii*3](net[0+ii*3](x)))
            x = torch.flip(x, [1])
        x = net[-1](x)
        return x

    def forward1(self, x):
        outs = []
        # x = torch.flip(x, [1])
        net = list(self._models[0].children())
        x = net[2](net[1](net[0](x)))
        outs.append(x.cpu().data)
        # x = torch.flip(x, [1])
        for ii in range(1, self.num_layers):
            x = net[2+ii*3](net[1+ii*3](net[0+ii*3](x)))
            outs.append(x.cpu().data)
            # x = torch.flip(x, [1])
        x = net[-1](x)
        outs.append(x.cpu().data)
        return outs

class vgg16_1(nn.Module):
    def __init__(self, num_bits):
        super(vgg16_1, self).__init__()
        net = tv.models.vgg16_bn(pretrained=True)
        self.net = nn.Sequential(*list(net.features.children())[:4])
        self.bits = num_bits
    def forward(self, x):
        out = self.net(x)
        return out[:, :self.bits, :, :]

class vgg16_2(nn.Module):
    def __init__(self, num_bits):
        super(vgg16_2, self).__init__()
        net = tv.models.vgg16_bn(pretrained=True, progress=True)
        self.net = nn.Sequential(*list(net.features.children())[:11])
        self.bits = num_bits
    def forward(self, x):
        b,_,h,w = x.shape
        out = self.net(x)
        out = out[:, :self.bits * 4, :, :]
        out = out.reshape([b, self.bits, h, w])
        return out



class BasicDecoderJpeg(nn.Module):
    def __init__(self, num_bits, hidden_size, num_layers, kernel_size, last_norm):
        super(BasicDecoderJpeg, self).__init__()
        self.jpeg = DiffJPEG.DiffJPEG(height=448, width=512, differentiable=True, quality=args.quality)
        self.decoder = BasicDecoder(num_bits, hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size,
                                    last_norm=last_norm)
    def forward(self, x):
        x = self.jpeg(x)
        return self.decoder(x).contiguous()

def load_model(num_bits, hidden_size, num_layers, kernel_size=3, last_norm=False, do_jpg=False):
    model = BasicDecoder(num_bits, hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size, last_norm=last_norm)
    model.apply(shuffle_params)

    model.to('cuda')
    return model


def load_image(idx, small_size=False):
    image = f"{args.image_path}/{idx+800:04d}.jpg"
    image = np.array(Image.open(image)).astype(np.float32) / 255.0
    image = image[:504, :, :]
    # if small_size:
    #     image = image[100:420, 100:408, :]
    # if args.smaller_size:
    #     image = image[100:260, 100:254, :]
    # if args.do_jpg:
    #     image = image[:448, :, :]
    image = torch.from_numpy(image).permute(2, 0, 1)[None].contiguous()
    image = image.to('cuda')
    return image


def load_target(image, bits):
    h, w = image.shape[-2:]
    if args.model=='model1':
        h = h // 4
        w = w // 4
        bits = bits * 16
    target = torch.bernoulli(torch.empty([1, bits, h, w]).uniform_(0, 1)).to(image.device)
    return target

import hamming_torch
# def load_hamming(image, bits):
#     h, w = image.shape[-2:]
#     target = torch.bernoulli(torch.empty([1, bits, h, w]).uniform_(0, 1)).cuda()
#     return target
#     # h74 = hamming_torch.Hamming74()
#     # h_target = h74.encode(target.view(-1, 4))
#     # return h_target.cuda()





criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').cuda()
criterion1 = torch.nn.L1Loss(reduction='sum').cuda()
criterion2 = torch.nn.MSELoss(reduction='sum').cuda()
criterion_none = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()


def get_loss(outputs, target, loss_mode):
    hinge = 0.3
    loss = None
    if loss_mode == "BCE":
        # import  IPython
        # import sys
        # IPython.embed()
        # sys.exit()
        loss = criterion(outputs, target)
    elif loss_mode == "BCE_weight":
        loss1 = criterion(outputs, target)
        loss2 = bit_loss(outputs, target)
        loss = loss1 + loss2 * 10
    elif loss_mode == "BCE_bit":
        loss1 = criterion_none(outputs, target)
        loss = new_loss(outputs, target, loss1)
    elif loss_mode == "BCE_bit1":
        loss1 = criterion_none(outputs, target)
        loss = new_loss1(outputs, target, loss1)
    elif loss_mode == "BCE_bit_vk":
        loss1 = criterion_none(outputs, target)
        loss = new_loss_vk(outputs, target, loss1)
    elif loss_mode == "BCE_bit3":
        loss1 = criterion_none(outputs, target)
        loss = new_loss3(outputs, target, loss1)
    elif loss_mode == "BCE_bit4":
        loss1 = criterion_none(outputs, target)
        loss = new_loss4(outputs, target, loss1)
    elif loss_mode == "BCE_bit5":
        loss1 = criterion_none(outputs, target)
        loss = new_loss5(outputs, target, loss1)
    elif loss_mode == "log":
        loss = -(target * 2 - 1) * outputs
        loss = torch.nn.functional.softplus(loss)  # log(1+exp(x))
        loss = torch.sum(loss)
    elif loss_mode == "hingelog":
        loss = -(target * 2 - 1) * outputs
        loss = torch.nn.functional.softplus(loss)  # log(1+exp(x))
        loss = torch.max(loss - hinge, torch.zeros(target.shape).to(target.device))
        loss = torch.sum(loss)
    elif loss_mode == "L1":
        outputs = F.sigmoid(outputs) * 255
        loss = criterion1(outputs, target)
    elif loss_mode == "L2":
        outputs = F.sigmoid(outputs) * 255
        loss = criterion2(outputs, target)
    return loss


# class outData():
#     def __init__(self, image, adv_image, adv_image1, acc1, acc2, bits_count1,
#                  bits_count2, target, out1, out2):
#         self.image = image
#         self.adv_image = adv_image
#         self.adv_image1 = adv_image1
#         self.acc1 = acc1
#         self.acc2 = acc2
#         self.bits_count1 = bits_count1
#         self.bits_count2 = bits_count2
#         self.target = target
#         self.out1 = out1
#         self.out2 = out2


if __name__ == '__main__':
    result_list = {}
    steps1 = args.steps1
    steps2 = args.steps2
    eps = args.eps
    max_iter = args.max_iter
    bits = args.bits
    lr1 = args.lr1
    lr2 = args.lr2
    loss1 = 'BCE'
    loss2 = args.loss2
    hm = hamming_torch.Hamming74('cuda:0')
    os.makedirs(args.save_path, exist_ok=True)
    for idx in tqdm.tqdm(range(args.start_idx, args.image_range + 1)):
        random.seed(idx)
        torch.manual_seed(idx)
        np.random.seed(idx)

        image = load_image(idx, small_size=args.small_size)
        h_bits = math.ceil(bits * 7 / 4)
        target = load_target(image, bits).view(-1)
        h_target = hm.encode(target.view(-1)).float()
        h_target = h_target.view(-1)
        lh = h_target.shape[0]
        adv_image = image.clone().detach()
        adv_image1 = image.clone().detach()
        # accuracy1 = []
        # accuracy2 = []
        flag = 0
        number = 0
        min_acc = 1
        target_decode1=None
        target_decode2=None
        acc1_list = []
        acc2_list = []
        bit_code_1 = []
        bit_code_2 = []
        while flag == 0 and number < args.number:
            number += 1
            random.seed(1111*number)
            torch.manual_seed(1111*number)
            np.random.seed(1111*number)
            model = load_model(h_bits, args.hidden, args.num_layers, args.kernel_size, args.last_norm,
                               do_jpg=args.do_jpg)
            adv_image = image.clone().detach()
            adv_image1 = image.clone().detach()
            for i in tqdm.tqdm(range(steps1 // max_iter)):
                adv_image.requires_grad = True
                optimizer = LBFGS([adv_image], lr=lr1, max_iter=max_iter)


                def closure():
                    outputs = model(adv_image)
                    loss = get_loss(outputs.view(-1)[:lh], h_target, loss1)

                    optimizer.zero_grad()
                    loss.backward()
                    return loss


                optimizer.step(closure)


                delta = torch.clamp(adv_image - image, min=-eps, max=eps)
                adv_image = torch.clamp(image + delta, min=0, max=1).detach()
                if not args.not_early:
                    with torch.no_grad():
                        out1 = (model(adv_image).view(-1)[:lh] > 0).float()
                        out1 = acc_bits(out1, h_target)
                        if out1[:6].sum() == 0:
                            print(out1, 0)
                            acc1_list.append(0)
                            bit_code_1.append(out1.cpu().data.numpy())
                            flag = 1
                            break
            if flag == 1:
                break
            adv_image1 = adv_image.clone().detach()
            for i in tqdm.tqdm(range(steps2 // max_iter)):
                adv_image1.requires_grad = True
                optimizer = LBFGS([adv_image1], lr=lr2, max_iter=max_iter)

                def closure():
                    outputs = model(adv_image1)
                    loss = get_loss(outputs.view(-1)[:lh], h_target, loss2)
                    optimizer.zero_grad()
                    loss.backward()
                    return loss


                optimizer.step(closure)
                delta = torch.clamp(adv_image1 - image, min=-eps, max=eps)
                adv_image1 = torch.clamp(image + delta, min=0, max=1).detach()
                if not args.not_early:
                    with torch.no_grad():
                        out1 = ((model(adv_image1).int().float()).view(-1)[:lh] > 0).float()
                        out1 = acc_bits(out1, h_target)
                        if out1[:6].sum() == 0:
                            print(out1, 1)
                            flag = 1
                            acc2_list.append(0)
                            bit_code_2.append(out1.cpu().data.numpy())
                            break
            if flag == 1:
                break
            with torch.no_grad():
                # if args.quant_last:
                #     adv_image = torch.clamp(adv_image * 255, 0, 255).int().float() / 255.
                #     adv_image1 = torch.clamp(adv_image1 * 255, 0, 255).int().float() / 255.
                out1 = (model(adv_image) > 0).float().view(-1)[:lh]
                out2 = (model(adv_image1) > 0).float().view(-1)[:lh]
                bout1 = acc_bits(out1, h_target)
                bout2 = acc_bits(out2, h_target)
                target_decode1 = hm.decode(out1.view(-1, 7)).float()
                target_decode2 = hm.decode(out2.view(-1, 7)).float()
                acc1 = (target_decode1.view(-1) != target.view(-1)).float().mean().item()
                acc2 = (target_decode2.view(-1) != target.view(-1)).float().mean().item()
                acc1_list.append(acc1)
                acc2_list.append(acc2)
                bit_code_1.append(bout1.cpu().data.numpy())
                bit_code_2.append(bout2.cpu().data.numpy())
                # # target_decode1 = target_decode1.cpu().data
                # # target_decode2 = target_decode2.cpu().data
                # min_acc = min([min_acc, acc1, acc2])
        adv_image = adv_image.cpu().squeeze().permute(1, 2, 0)
        adv_image1 = adv_image1.cpu().squeeze().permute(1, 2, 0)
        image = image.cpu().squeeze().permute(1, 2, 0)
        print(acc1_list, bit_code_1)
        print(acc2_list, bit_code_2)
        with open(f'{args.save_path}/acc.txt', 'a') as f:
            f.write(f'{idx:04d}.png\n')
            for a1 in acc1_list:
                f.write(f'{a1:.8f} ')
            f.write('\n')
            for a1 in acc2_list:
                f.write(f'{a1:.8f} ')
            f.write('\n')
        with open(f'{args.save_path}/bit_acc.txt', 'a') as f:
            f.write(f'{idx:04d}.png\n')
            for iii, a1 in enumerate(bit_code_1):
                f.write(f'{iii}: {str(a1)}\n')
            for iii, a2 in enumerate(bit_code_2):
                f.write(f'{iii}: {str(a2)}\n')
            f.write('\n')

        result = (image, adv_image, adv_image1)
        with open(f'{args.save_path}/{idx:04d}.pkl', 'wb') as f:
            pickle.dump(result, f)
        # print(acc1, acc2, min_acc)
        # result_list[idx] = result
    # folder = os.path.dirname(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    # with open(f'{args.save_path}/result.pkl','wb') as f:
    #     pickle.dump([result_list, args], f)
