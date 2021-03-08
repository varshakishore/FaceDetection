import argparse
import math
import os
import time

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

import lpips
from model import Generator

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image generator from latent vector"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to the checkpoint"
    )
    
    args = parser.parse_args()
    
    n_mean_latent = 10000
    
    torch.manual_seed(0)
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load("stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
    g_ema.eval()
    
    g_ema = g_ema.to("cuda")
    
    
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    
    result_file = torch.load(args.checkpoint)
    
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(len(result_file.keys()), 1, 1, 1).normal_())

    torch.manual_seed(1)
    
    for input_name in result_file.keys():
        latent = result_file[input_name]["latent"]
        saved_noise = result_file[input_name]["noise"]
        img_gen = result_file[input_name]["img"]
        img_regenerate = g_ema([latent.unsqueeze(0)], input_is_latent=True, noise=saved_noise)
        img_regenerate_noise_check = g_ema([latent.unsqueeze(0)], input_is_latent=True, noise=noises)
        print(torch.abs(img_gen - img_regenerate[0]).mean())
        print(torch.abs(img_gen - img_regenerate_noise_check[0]).mean())
        
    
