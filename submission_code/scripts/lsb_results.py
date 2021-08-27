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
parser.add_argument('--pretrained', action="store_true")
parser.add_argument('--psnr_weight', type=float, default=0.5)


args = parser.parse_args()
print(args.name)
expname = "main100_1"


# models
pretrained = args.pretrained
flatten_image = False
steps = 2000
max_iter = 20
alpha = 0.1
eps = 0.105
psnr_weight = args.psnr_weight


def psnr_loss(img1, img2):
    weight = torch.tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256.0
    diff = torch.sum((img1 - img2) * weight.to(img1.device), dim=1)
    return 10 * torch.log10(torch.norm(diff))


for seed in [11111]:

    image = args.dataset_path + args.name
    image = imread(image, pilmode='RGB')
    image = torch.LongTensor(image).permute(2, 1, 0).unsqueeze(0)

    torch.manual_seed(int(args.name[:-4]))
    target = torch.bernoulli(torch.empty(image.shape).uniform_(0, 1)).to(image.device).long()
    print(target.shape)

    adv_image = (image & 0b11111110) | target

    print(seed)
    psnr = calc_psnr((image.squeeze().permute(2,1,0).float()).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0).float()).detach().cpu().numpy())
    print("psnr:", psnr)
    print("ssim:", calc_ssim((image.squeeze().permute(2,1,0).float()).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0).float()).detach().cpu().numpy()))
    # print("error:", acc)
    lbfgsimg = (adv_image.cpu().squeeze().permute(2,1,0).numpy()).astype(np.uint8)
    if psnr > 20:
        break

os.makedirs(args.output_image_path, exist_ok = True)
imname = args.output_image_path+f'{args.num_bits}_{expname}_{args.name[:-4]}.jpg'
Image.fromarray(lbfgsimg).save(imname, "JPEG", quality=0)

image_read = imread(imname, pilmode='RGB')
image_read = torch.LongTensor(image_read).permute(2, 1, 0).unsqueeze(0)

acc = torch.sum((image_read & 1) ^ target).float() / target.numel()
# acc = len(torch.nonzero((model(image_read)>0).float().view(-1) != target.view(-1))) / target.numel()
print("read:", f"{acc*100:0.2f}%")
psnr = calc_psnr((image.squeeze().permute(2,1,0).float()).detach().cpu().numpy(), (image_read.squeeze().permute(2,1,0).float()).detach().cpu().numpy())
ssim = calc_ssim((image.squeeze().permute(2,1,0).float()).detach().cpu().numpy(), (image_read.squeeze().permute(2,1,0).float()).detach().cpu().numpy())
print("psnr", psnr)
print("ssim", ssim)

# with open('final_exps_r.csv', mode='a') as file:
#     file.write(f'{args.num_bits}, {expname}, {args.name}, {eps}, {seed}, {end - start}, {psnr}, {ssim}, {acc} \n')





