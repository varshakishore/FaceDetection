import sys, os, argparse, math

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torch.optim as optim

import datasets, hopenet, utils
from models.stylegan_generator import StyleGANGenerator

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='/output/snapshots/AFLW2000/_epoch_9.pkl', type=str)
    parser.add_argument('--target_file', help='target to match.', type=str)
    parser.add_argument('--output_dir', help='directory to save results', type=str)
    parser.add_argument('--w_file', help='random w.', type=str)

    args = parser.parse_args()

    return args

def model_frozen(model):
    for para in model.parameters():
        para.requires_grad = False

def get_pose(image, model, idx_tensor):
    yaw,yaw_1,yaw_2,yaw_3,yaw_4, pitch,pitch_1,pitch_2,pitch_3,pitch_4, roll,roll_1,roll_2,roll_3,roll_4 = model(image)

    # Binned predictions
    _, yaw_bpred = torch.max(yaw, 1)
    _, pitch_bpred = torch.max(pitch, 1)
    _, roll_bpred = torch.max(roll, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw, 1)
    pitch_predicted = utils.softmax_temperature(pitch, 1)
    roll_predicted = utils.softmax_temperature(roll, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) - 99
    
    return yaw_predicted, pitch_predicted, roll_predicted
    
def gan_forward(z_tensor, model, latent_type="z"):
    if latent_type == "z":
        ws = model.model.mapping(z_tensor)
        wps = model.model.truncation(ws)
    elif latent_type == "w":
        ws = z_tensor
        wps = model.model.truncation(ws)
    elif latent_type == "ws":
        wps = z_tensor
    images = model.model.synthesis(wps)
    return images
    
def postprocess(images, scale=1.0):
    images = (images + 1) * 255 / 2
    images = torch.clamp(images + 0.5, 0, 255) * scale / 255
    return images
    
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Multinet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 198)
    model.cuda(gpu)
    
    transformations_from_gan = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    
    print('Loading GAN.')
    GAN_model = StyleGANGenerator("stylegan_ffhq")
    GAN_model.model = GAN_model.model.to("cuda")
    
    print('Loading pose direction.')
    pose_dir = torch.from_numpy(np.load("../interfacegan/boundaries/stylegan_ffhq_pose_w_boundary.npy")).to("cuda").float().squeeze()
    
    print('Loading target.')
    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_img = transformations(Image.open(args.target_file).convert("RGB")).to("cuda").unsqueeze(0)
    
    print('Loading w')
    ws = np.load(args.w_file)
    
    print('Ready to test network.')
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    model_frozen(model)
    model_frozen(GAN_model.model)
    total = 0

    idx_tensor = [idx for idx in range(198)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    print("compute target")
    with torch.no_grad():
        yaw_target, pitch_target, roll_target = get_pose(target_img, model, idx_tensor)
        yaw_target = -(yaw_target * math.pi / 180)
        print(f"target yaw: {yaw_target}")

    for i, w in tqdm(enumerate(ws), total=len(ws)):
        w_tensor = torch.from_numpy(w).to("cuda").unsqueeze(0).float()
        w_tensor = w_tensor - w_tensor.matmul(pose_dir).to("cuda") * pose_dir
        epochs = 10
        
        def forward(eps):
            image = gan_forward(w_tensor + eps * pose_dir, GAN_model, latent_type="w")
            image = postprocess(image)
            transformed_image = transformations_from_gan(image)
            yaw_predicted, pitch_predicted, roll_predicted = get_pose(transformed_image, model, idx_tensor)
            return -(yaw_predicted * math.pi / 180)
        
        eps_min = torch.tensor([5]).float().to("cuda")
        eps_max = torch.tensor([-5]).float().to("cuda")
        for epoch in range(epochs):
            eps = (eps_min + eps_max) / 2
            yaw_predicted = forward(eps)
            if yaw_predicted > yaw_target:
                eps_min = eps
            else:
                eps_max = eps
        print(f"eps: {eps.item()}, yaw_predicted: {yaw_predicted.item()}, yaw_target: {yaw_target.item()}, loss: {torch.abs(yaw_predicted - yaw_target).item()}")
        image = gan_forward(w_tensor + eps * pose_dir, GAN_model, latent_type="w")
        image = postprocess(image)
        plt.imsave(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.target_file))[0] + f"_{i}_pose_match.jpg"), image.permute(0, 2, 3, 1).squeeze().cpu().numpy())
        
