import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import numpy as np

import datasets, hopenet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='/AFLW2000/', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='/output/snapshots/AFLW2000/_epoch_9.pkl', type=str)
    parser.add_argument('--mask_dir', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Multinet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 198)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.AFLW2000_BYL(args.data_dir, transformations)
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size=args.batch_size, num_workers=2)

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(198)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    save_pose_vectors = []
    # end = time.time()
    for i, data in enumerate(test_loader):
        images, names = data
        images = Variable(images).cuda(gpu)
        yaw,yaw_1,yaw_2,yaw_3,yaw_4, pitch,pitch_1,pitch_2,pitch_3,pitch_4, roll,roll_1,roll_2,roll_3,roll_4 = model(images)

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() - 99

        save_pose_vectors.append([names[0], yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item()])

        if i % 1000 == 0:
            print(i // 1000)
            # print(time.time() - end)
            # end = time.time()
            # import pdb; pdb.set_trace()

        # if (i + 1) % 10000 == 0:
        #     np.save('pose_info_{}.npy'.format((i + 1) // 10000), save_pose_vectors)
        #     save_pose_vectors = []
    np.save('{}/pose_info.npy'.format(args.data_dir), save_pose_vectors)

    # np.load('{}/pose_info.npy'.format(args.data_dir))
    # import pdb; pdb.set_trace()

    # # save skin colors
    # masks = np.load(args.mask_dir)
    # save_color_vectors = [] 
    # # torch.from_numpy(np.load(args.mask)).to("cuda")
    # for i, images in enumerate(test_loader):
    #     mask = masks[i]
    #     img = images * mask
    #     m_value = np.mean(img)
    #     save_color_vectors.append(m_value)
    
    # np.save('color_info.npy', save_color_vectors)

    # import pdb; pdb.set_trace()

        
