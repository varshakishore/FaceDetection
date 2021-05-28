import numpy as np
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import argparse
import torch
import torchvision
from torch.optim import LBFGS
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
import cv2


def shuffle_params(m):
    if type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))
    if type(m) == nn.BatchNorm2d:
        if "track_running_stats" in m.__dict__:
            m.track_running_stats = False


def shuffle(t):
    idx = torch.randperm(t.numel())
    t = t.view(-1)[idx].view(t.shape)
    return t


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
            self._conv2d(self.divstack * self.divstack * (1 if self.flatten_image else 3), self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer() if self.yan_norm else nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer() if self.yan_norm else nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer() if self.yan_norm else nn.BatchNorm2d(self.hidden_size),
#             self._conv2d(self.hidden_size, 2)
            self._conv2d(self.hidden_size, self.data_depth * self.divstack * self.divstack // 3 if self.flatten_image else self.data_depth * self.divstack * self.divstack)
        )

        return [self.layers]

    def __init__(self, data_depth, hidden_size, yan_norm=False, divstack=1, flatten_image=False):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.yan_norm = yan_norm
        self.divstack = divstack
        self.flatten_image = flatten_image

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


criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
criterion1 = torch.nn.L1Loss(reduction='sum')
criterion2 = torch.nn.MSELoss(reduction='sum')


def pad(x, reweight=7):
    return torch.cat(
        [x.view(-1), torch.zeros(reweight - 1 - (x.numel() - 1) % reweight).to(x.dtype).to(x.device)]).view(-1,
                                                                                                            reweight)


def get_loss(model, adv_image, target, loss_mode, hinge, reweight=0):
    outputs = model(adv_image)
    if loss_mode == "BCE":
        loss = criterion(outputs, target)
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

    if reweight > 0:
        correctness = (outputs > 0).long() != target.long()
        correctness = pad(correctness, reweight)
        loss_mask = (torch.sum(correctness, dim=1) > 1).float().view(-1, 1)
        loss = torch.sum(pad(loss, reweight) * loss_mask)
        loss.backward()
    return loss


def flatten(image, dim=3):
    img = torch.cat(torch.split(image, 1, dim=1), dim=dim)
    return img


parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--unroll', type=str, default="on", choices=["on", "off"])
parser.add_argument('--pgd', type=str, default="on", choices=["on", "off"])
args = parser.parse_args()


if __name__ == "__main__":
    divstack = 1
    yan_norm = True
    shuffle_image = False
    flatten_image = args.unroll == "on"
    loss_mode = "log"
    hinge = 0.3
    steps = 1000
    eps = 0.2
    alphas = [0.3, 0.2, 0.1]
    max_iter = 20
    pgd = args.pgd == "on"
    # val_root = "/home/vk352/FaceDetection/datasets/div2k/val/_"
    val_root = "/home/vk352/FaceDetection/datasets/div2k/val/512"

    out_root = os.path.join(f"{'unroll_' if flatten_image else ''}{args.bits}bits_dim{args.dim}{'_pgd' if pgd else ''}")
    os.makedirs(out_root, exist_ok=True)
    # models

    model = BasicDecoder(args.bits, hidden_size=args.hidden_size, yan_norm=yan_norm, divstack=divstack, flatten_image=flatten_image)
    model.apply(shuffle_params)
    model.to('cuda')#.half()

    for fname in os.listdir(val_root):
        image = os.path.join(val_root, fname)
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to('cuda')#.half()
        if divstack > 1:
            image = pad(image, divstack)
        if shuffle_image:
            image = shuffle(image)
        if flatten_image:
            image = flatten(image, args.dim)

        torch.manual_seed(int(fname[:-4]))
        target = torch.bernoulli(torch.empty([1, args.bits//3 if flatten_image else args.bits, image.shape[2], image.shape[3]]).uniform_(0, 1)).to(image.device)#.half()

        adv_image = image.clone().detach()
        print(fname, image.shape)

        accs = []

        for alpha in alphas:
            for i in trange(steps // max_iter // len(alphas)):
                adv_image.requires_grad = True
                if args.pgd:
                    loss = get_loss(model, adv_image, target, loss_mode, hinge)
                    grad = torch.autograd.grad(loss, adv_image, retain_graph=False, create_graph=False)[0]
                    adv_image = adv_image.detach() - alpha * grad.sign()
                else:
                    optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)
                    def closure():
                        loss = get_loss(model, adv_image, target, loss_mode, hinge)
                        optimizer.zero_grad()
                        loss.backward()
                        return loss
                    optimizer.step(closure)

                delta = torch.clamp(adv_image - image, min=-eps, max=eps)
                adv_image = torch.clamp(image + delta, min=0, max=1)
                adv_image = torch.clamp(adv_image * 255, 0, 255).int().float() / 255.
                adv_image = adv_image.detach()

                if loss_mode in ["L1", "L2"]:
                    acc = len(torch.nonzero(torch.abs(F.sigmoid(model(adv_image)).float().view(-1) * 255 - target.view(-1)) > 128)) / target.numel()
                else:
                    acc = len(torch.nonzero((model(adv_image) > 0).float().view(-1) != target.view(-1))) / target.numel()
                print("error", acc)
                accs.append(acc)
        with open(os.path.join(out_root, "error.txt"), "a") as f:
            f.write(f"{fname}, {min(accs) * 100}\n")
        img = torch.cat(torch.split(adv_image, 512, dim=args.dim), dim=1).detach()
        img = img.cpu().squeeze(0).permute(2, 1, 0).numpy()
        img = (img*255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_root, fname), img)


