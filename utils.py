from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from skimage import io

class getLfwDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, device, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample. mtcnn is used otherwise
            
        """
        self.df = pd.read_csv(csv_file, sep='\t')
        self.root_dir = root_dir
        self.mtcnn = MTCNN(image_size=182, margin=44, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                           device=device)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        img_name = os.path.join(self.root_dir, str(row['Name']) + '/'+ str(row['Name'])+ f'_0001.jpg')
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.mtcnn(image)
                                
        sample = {'image': image}

        return sample
    
def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class NumpyResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        r"""
        Args:
            img (np array): image to be resized
        Returns:
            np array: resized image
        """
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        print("resize done")
        return np.array(img.resize(self.size, resample=Image.BILINEAR))

    def __repr__(self):
        return self.__class__.__name__ 
    
class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        r"""
        Turn a numpy objevt into a tensor.
        """
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        return Transforms.functional.to_tensor(img)
    