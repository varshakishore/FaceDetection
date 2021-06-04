import os
import numpy as np
import torch
import torch.nn as nn
from imageio import imread
from PIL import Image


def flatten(image, dim=3):
    img = torch.cat(torch.split(image, 1, dim=1), dim=dim)
    return img


def shuffle_params(m):
    if type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


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
            self._conv2d(1 if self.flatten_image else 3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            normLayer(),

            self._conv2d(self.hidden_size,
                         self.data_depth // 3 if self.flatten_image else self.data_depth)
        )

        return [self.layers]

    def __init__(self, data_depth, hidden_size, flatten_image=False):
        super(BasicDecoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.flatten_image = flatten_image
        self._models = self._build_models()

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x


def load_image_and_target(fname, num_bits, flatten_image, seed=None, device='cuda'):
    image = imread(fname, pilmode='RGB') / 255.0
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if flatten_image:
        image = flatten(image, dim=3)

    index = os.path.basename(fname).split(".")[-2]
    if index.isdigit():
        seed = int(index)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Set random seed to {seed}")

    target = torch.bernoulli(torch.empty([1, num_bits // 3 if flatten_image else num_bits, image.shape[2], image.shape[3]]).uniform_(0, 1)).to(device)

    return image, target


def save_image(img, fname):
    img = (img.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(fname)
    print(f"image has been saved to {fname}")
