import numpy as np
import os
import torch
import argparse
from steganogan import SteganoGAN
from steganogan.loader import DataLoader
from steganogan.encoders import BasicEncoder, DenseEncoder
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.critics import BasicCritic
from tqdm import tqdm
from imageio import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--eval', action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    train = DataLoader('/home/vk352/FaceDetection/datasets/div2k/train/', limit=np.inf, shuffle=True, batch_size=4)
    validation = DataLoader('/home/vk352/FaceDetection/datasets/div2k/val/', limit=np.inf, shuffle=True, batch_size=1 if args.eval else 4)
    save_path = f'{args.bits}_bits.steg'
    out_folder = f"{args.bits}_bits"

    if os.path.isfile(save_path):
        steganogan = SteganoGAN.load(architecture=None, path=save_path, cuda=True, verbose=True)
    else:
        steganogan = SteganoGAN(args.bits, BasicEncoder, BasicDecoder, BasicCritic, hidden_size=args.hidden_size, cuda=True, verbose=True)
    #     steganogan.save(save_path)
    print(steganogan.data_depth)

    errs = []
    if args.eval:
        os.makedirs(out_folder, exist_ok=True)
        for i, (cover, _) in tqdm(enumerate(train)):
            cover = cover.cuda()
            generated, payload, decoded = steganogan._encode_decode(cover)

            generated = (generated[0].clamp(-1.0, 1.0).permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
            imwrite(os.path.join(out_folder, f"{i:04d}.png"), generated.astype('uint8'))
            error = 1 - (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
            print(i, error)
            errs.append(error)
            with open(os.path.join(out_folder, f"err.txt"), "a") as f:
                f.write(f"{error}\n")
        print(np.mean(errs) * 100.0)
    else:
        steganogan.fit(train, validation, save_path, epochs=100)
