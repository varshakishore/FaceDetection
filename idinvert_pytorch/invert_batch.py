#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
from tqdm import tqdm
import numpy as np
from os import listdir, walk
from os.path import join

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image
import argparse



# In[2]:


model_name = 'styleganinv_ffhq256'
output_dir = ''
learning_rate = 0.01
num_iterations = 70
num_results = 1
loss_weight_feat = 5e-5
loss_weight_enc = 2.0
viz_size = 0
visualizer = False
gpu_id = "0"


parser = argparse.ArgumentParser()
parser.add_argument("-flip", "--flip", help="flip directory order",
                    action="store_true")
parser.add_argument("-part", "--part", help="which part",
                    type=int)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

inverter = StyleGANInverter(
  model_name,
  learning_rate=learning_rate,
  iteration=num_iterations,
  reconstruction_loss_weight=1.0,
  perceptual_loss_weight=loss_weight_feat,
  regularization_loss_weight=loss_weight_enc)
image_size = inverter.G.resolution



  # Invert images.
path = '/home/vk352/FaceDetection/datasets/celeba/Img_cropped/train'
path_gan = '/home/vk352/FaceDetection/datasets/celeba/gan_id'

folders = sorted(listdir(path))
# if args.flip:
#     folders = folders[::-1]
div = len(folders)//4
folders = folders[args.part*div:(args.part+1)*div]
folders = folders[::-1]

for folder in folders:
    folders_gan = listdir(path_gan)
    if folder in folders_gan:
        continue
    latent_codes = []
    print(folder)
    output_dir = f'/home/vk352/FaceDetection/datasets/celeba/gan_id/{folder}'
    os.makedirs(output_dir, exist_ok=True) 
    for (dirpath, dirnames, filenames) in walk(join(path, folder)):
        filenames = sorted(filenames)

        for img_idx in tqdm(range(len(filenames)), leave=False):
    #         image_path = image_list[img_idx]
            image_name = filenames[img_idx]
            image = resize_image(load_image(f"{dirpath}/"+image_name), (image_size, image_size))
            code, viz_results = inverter.easy_invert(image, num_viz=num_results)
            latent_codes.append(code)
        #     save_image(f'{output_dir}/{image_name}_ori.png', image)
        #     save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
            save_image(f'{output_dir}/{image_name[:-4]}_inv.jpg', viz_results[-1])
#             if visualizer:
#                 visualizer.set_cell(img_idx, 0, text=image_name)
#                 visualizer.set_cell(img_idx, 1, image=image)
#                 for viz_idx, viz_img in enumerate(viz_results[1:]):
#                     visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)

      # Save results.
    if (len(latent_codes)==1):
        np.save(f'{output_dir}/inverted_codes.npy', np.expand_dims(latent_codes, axis=0))
    elif (len(latent_codes)>1):
        np.save(f'{output_dir}/inverted_codes.npy', np.concatenate(latent_codes, axis=0))

#     if visualizer:
#         visualizer.save(f'{output_dir}/inversion.html')


# In[8]:





# In[ ]:




