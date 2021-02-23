import os
from os import listdir, walk
from os.path import join
from tqdm import tqdm
path = '/home/vk352/FaceDetection/datasets/celeba/Img_cropped/train'
path_gan = '/home/vk352/FaceDetection/datasets/celeba/gan'
folders_gan = listdir(path_gan)
folders = sorted(listdir(path))
div = len(folders)//4
folders = folders[3*div:][::-1]
for folder in folders:
    if folder in folders_gan:
        continue
    for (dirpath, dirnames, filenames) in walk(join(path, folder)):
        print(folder)
        out_folder = f"/home/vk352/FaceDetection/datasets/celeba/gan/{folder}/"
        os.makedirs(out_folder, exist_ok=True) 
        filenames = sorted(filenames)[:10]
        for i_file in range(0, len(filenames), 5):
            files = ' '.join([f"{dirpath}/"+elem for elem in filenames[i_file:i_file+5]])
            print(f"python projector.py --ckpt stylegan2-ffhq-config-f.pt --size=1024 --step=350 --output_folder={out_folder} {files}")
            os.system(f"python projector.py --ckpt stylegan2-ffhq-config-f.pt --size=1024 --step=350 --output_folder={out_folder} {files}")
