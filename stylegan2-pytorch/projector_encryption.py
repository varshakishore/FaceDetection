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


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument(
        "--latent_regularize",
        type=float,
        default=1e5,
        help="weight of the latent regularization when latent is z",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument("--p", type=float, default=1.0, help="weight of the perceptual loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--output_folder", type=str, default="results/", help="path to the output"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="the mask range to mask out the face [w1, w2, h1, h2]",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )
    parser.add_argument('--input_is_not_latent', action='store_true', default=False)
    

    args = parser.parse_args()
    
#     img_name = args.output_folder + os.path.splitext(os.path.basename(args.files[0]))[0] + "-project.jpg"
#     if os.path.exists(img_name):
#         print("already projected: {}".format(img_name))
#         exit()
#     else:
#         print("generating: {}".format(img_name))

    n_mean_latent = 10000
    
    input_is_latent = not args.input_is_not_latent

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in args.files:
#         "/home/vk352/FaceDetection/datasets/celeba/Img_cropped/test/"
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)
    
    torch.manual_seed(0)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    
    for param in g_ema.parameters():
        param.requires_grad = False
    
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    torch.manual_seed(args.seed)
        
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = False

    #optimizer = optim.Adam([latent_in] + noises, lr=args.lr)
    optimizer = optim.Adam([latent_in], lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []
    
    base_size = 256
    if args.mask is not None:
        mask = torch.zeros(imgs.shape).to(device)
        h1, h2, w1, w2 = args.mask.split("_")
        h1, h2, w1, w2 = int(float(h1) / args.size * base_size), int(float(h2) / args.size * base_size), int(float(w1) / args.size * base_size), int(float(w2) / args.size * base_size)
        mask[:, :, :w1] = 1
        mask[:, :, w2:] = 1
        mask[:, :, :, :h1] = 1
        mask[:, :, :, h2:] = 1
        imgs = imgs * mask

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        #latent_n = latent_noise(latent_in, noise_strength.item())
        latent_n = latent_in
        img_gen, _ = g_ema([latent_n], input_is_latent=input_is_latent, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

            
        if args.mask is not None:
            mse_loss = F.mse_loss(img_gen * mask, imgs)
            if args.p == 0:
                p_loss = torch.zeros([1]).to("cuda")
            else:
                p_loss = percept(img_gen * mask, imgs).sum()
            if args.input_is_not_latent:
                reg_loss = torch.abs(torch.norm(latent_in) - math.sqrt(len(latent_in)))
            else:
                reg_loss = torch.zeros([1]).to("cuda")
#             if i%100 == 0:
#                 save_image(img_gen, args.output_folder + "/debug_img_gen_{}.jpg".format(i))
#                 save_image(img_gen * mask, args.output_folder + "/debug_img_gen_mask_{}.jpg".format(i))
#                 save_image(imgs, args.output_folder + "/debug_img_mask_{}.jpg".format(i))
            loss = args.p * p_loss + args.mse * mse_loss + args.latent_regularize * reg_loss
        else:
            p_loss = percept(img_gen, imgs).sum()
            mse_loss = F.mse_loss(img_gen, imgs)
            if args.input_is_not_latent:
                reg_loss = torch.abs(torch.norm(latent_in) - math.sqrt(len(latent_in)))
            else:
                reg_loss = torch.zeros([1]).to("cuda")
            loss = args.p * p_loss + args.mse * mse_loss + args.latent_regularize * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 10 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; regularize: {reg_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=input_is_latent, noise=noises)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
            "path": latent_path
        }

        if args.mask is not None:
            img_name = os.path.splitext(os.path.basename(input_name))[0] + "_mask_" + args.mask + "_{}-project.jpg".format(args.seed)
        else:
            img_name = os.path.splitext(os.path.basename(input_name))[0] + "_{}-project.jpg".format(args.seed)
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(args.output_folder+img_name)

    torch.save(result_file, args.output_folder+filename)
