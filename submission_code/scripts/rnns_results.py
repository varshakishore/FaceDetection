import numpy as np 
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import sys
import argparse
from PIL import Image
from utils import calc_psnr, calc_ssim
import os
import time

import torch
from torch.optim import LBFGS
import torch.nn.functional as F

import sys
sys.path.append("..")
# from steganogan.decoders import DenseDecoderNLayers
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan import SteganoGAN

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    help='file name')
parser.add_argument('--dataset_path', type=str,
                    help='path to dataset')
parser.add_argument('--num_bits', type=int,
                    help='number of bits to hide')
parser.add_argument('--output_image_path', type=str,
                    help='output image path')

args = parser.parse_args()
print(args.name)
expname = "main100_1"

def flatten(image, dim=3):
    img = torch.cat(torch.split(image, 1, dim=1), dim=dim)
    return img

def shuffle_params(m):
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())
        
        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))

class normLayer(nn.Module):
    def __init__(self):
        super(normLayer, self).__init__()
    
    def forward(self, x):
        b,c,h,w = x.shape
        assert b == 1
        mean = x.view(c, -1).mean(-1)
        std = x.view(c, -1).std(-1)
        x = x - mean.reshape([1, c, 1, 1])
        x = x / (std + 1e-7).reshape([1,c,1,1])
        return x



class BasicDecoder(nn.Module):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.layers = nn.Sequential(
            self._conv2d(self.divstack * self.divstack * (1 if flatten_image else 3), self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size, self.data_depth * self.divstack * self.divstack // 3 if flatten_image else self.data_depth * self.divstack * self.divstack)
        )

        return [self.layers]

    def __init__(self, data_depth, hidden_size):
        super(BasicDecoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.divstack = 1

        self._models = self._build_models()

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

hidden_size=128
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')


# models
pretrained = False
flatten_image = False
steps = 2000
max_iter = 20
alpha = 0.1
eps = 0.105

for seed in [11111,22222,33333,44444,55555,66666,777777,88888,99999,0]:
    if pretrained:
        steganogan = SteganoGAN.load(path=f"demo_{args.num_bits}.1.steg", cuda=True, verbose=True)
        model = steganogan.decoder
    else:
        np.random.seed(seed)
        model = BasicDecoder(args.num_bits, hidden_size=hidden_size)
        model.apply(shuffle_params)
        model.to('cuda')
    print(model)

    image = args.dataset_path + args.name
    image = imread(image, pilmode='RGB') / 255.0
    image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
    image = image[:, :, :, :]
    image = image.to('cuda')
    if flatten_image:
        image = flatten(image, 3)
    out = model(image)

    torch.manual_seed(int(args.name[:-4]))
    target = torch.bernoulli(torch.empty(out.shape).uniform_(0, 1)).to(out.device)
    print(target.shape)
    eps = eps-0.0005
    print("eps:", eps)
    adv_image = image.clone().detach()
    
    start = time.time()

    for i in range(steps // max_iter):
        adv_image.requires_grad = True
        optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)

        def closure():
            outputs = model(adv_image)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()

        acc = len(torch.nonzero((model(adv_image)>0).float().view(-1) != target.view(-1))) / target.numel()
#             print(i, acc)
        if acc==0: break
    end = time.time()
    print(seed)
    psnr = calc_psnr((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy())
    print("psnr:",psnr)
    print("ssim:",calc_ssim((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy()))
    print("error:", acc)
    lbfgsimg = (adv_image.cpu().squeeze().permute(2,1,0).numpy()*255).astype(np.uint8)
    if psnr>20: break

os.makedirs(args.output_image_path, exist_ok = True)
imname = args.output_image_path+f'{args.num_bits}_{expname}_{args.name[:-4]}.png'
Image.fromarray(lbfgsimg).save(imname)

image_read = imread(imname, pilmode='RGB') / 255.0
image_read = torch.FloatTensor(image_read).permute(2, 1, 0).unsqueeze(0).to('cuda')
acc = len(torch.nonzero((model(image_read)>0).float().view(-1) != target.view(-1))) / target.numel()
print("read:", acc)
psnr = calc_psnr((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (image_read.squeeze().permute(2,1,0)*255).detach().cpu().numpy())
ssim = calc_ssim((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (image_read.squeeze().permute(2,1,0)*255).detach().cpu().numpy())
print("psnr", psnr)
print("ssim", ssim)

with open('final_exps_r.csv', mode='a') as file:
    file.write(f'{args.num_bits}, {expname}, {args.name}, {eps}, {seed}, {end - start}, {psnr}, {ssim}, {acc} \n') 





