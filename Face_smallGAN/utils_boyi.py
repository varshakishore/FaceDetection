from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
from torchvision import datasets
import torch.utils.data as torchdata
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
from PIL import Image
import usps
import random


def get_mnist2svhn_data_loaders(conf):
    selected_labels = conf['selected_labels'] if 'selected_labels' in conf else None
    selected_scaled_labels = conf['selected_scaled_labels'] if 'selected_scaled_labels' in conf else None
    selected_labels_scale = conf['selected_labels_scale'] if 'selected_labels_scale' in conf else None
    if(conf['a2b'] == 1):
        train_loader_a = get_data_loader_mnist(conf['batch_size'], True, new_size=32, num_workers=4,
                                              selected_labels=selected_labels,
                                              selected_scaled_labels=selected_scaled_labels,
                                              selected_labels_scale=selected_labels_scale)
        test_loader_a = get_data_loader_mnist(conf['batch_size'], False, new_size=32, num_workers=4)
        train_loader_b = get_data_loader_svhn(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels)
        test_loader_b = get_data_loader_svhn(conf['batch_size'], False, new_size=32, num_workers=4)
    else:
        train_loader_a = get_data_loader_svhn(conf['batch_size'], True, new_size=32, num_workers=4,
                                              selected_labels=selected_labels,
                                              selected_scaled_labels=selected_scaled_labels,
                                              selected_labels_scale=selected_labels_scale)
        test_loader_a = get_data_loader_svhn(conf['batch_size'], False, new_size=32, num_workers=4)
        train_loader_b = get_data_loader_mnist(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels)
        test_loader_b = get_data_loader_mnist(conf['batch_size'], False, new_size=32, num_workers=4)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b



def get_mnist2usps_data_loaders(conf):
    selected_labels = conf['selected_labels'] if 'selected_labels' in conf else None
    selected_scaled_labels = conf['selected_scaled_labels'] if 'selected_scaled_labels' in conf else None
    selected_labels_scale = conf['selected_labels_scale'] if 'selected_labels_scale' in conf else None
    if(conf['a2b'] == 1):
        train_loader_a = get_data_loader_mnist(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels
                                               )
        test_loader_a = get_data_loader_mnist(conf['batch_size'], False, new_size=32, num_workers=4)
        train_loader_b = get_data_loader_usps(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels)
        test_loader_b = get_data_loader_usps(conf['batch_size'], False, new_size=32, num_workers=4)
    else:
        train_loader_a = get_data_loader_usps(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels)
        test_loader_a = get_data_loader_usps(conf['batch_size'], False, new_size=32, num_workers=4)
        train_loader_b = get_data_loader_mnist(conf['batch_size'], True, new_size=32, num_workers=4,
                                               selected_labels=selected_labels,
                                               selected_scaled_labels=selected_scaled_labels)
        test_loader_b = get_data_loader_mnist(conf['batch_size'], False, new_size=32, num_workers=4)


    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_data_loader_svhn(batch_size, train, new_size=32, num_workers=4, selected_labels=None, selected_scaled_labels=None, selected_labels_scale=0.5):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)
    if train:
        split = 'train'
    else:
        split = 'test'
    dataset = datasets.SVHN(root='datasets/svhn', download=True, transform=transform, split=split)

    if train:
        if selected_labels is not None:  # only use data with selected labels
            print('entering selected_table circle for svhn')
            index = torch.from_numpy(np.in1d(dataset.labels, selected_labels).astype(np.uint8))
        else:
            selected_labels = []
            index = torch.from_numpy(np.in1d(dataset.labels, selected_labels).astype(np.uint8))
        print(index.__len__())

        if selected_scaled_labels is not None:
            print('entering selected_scaled_labels circle for svhn')
            for scale_label in selected_scaled_labels:
                index1 = torch.from_numpy(np.in1d(dataset.labels, scale_label).astype(np.uint8))
                index1non = np.nonzero(index1)
                len_index1non = len(index1non)
                order = np.array(range(len_index1non))
                random.shuffle(order)
                for i in order[int(len_index1non * (1 - int(scale_label) * 0.1)):len_index1non]:
                    index1[index1non[i]] = 0
                index = index + index1
            # dataset.data = dataset.data[index]
            # dataset.labels = dataset.labels[index]
            print('entering rebuilding dataset circle *** src:svhn')
        svhndata = []
        svhnlabels = []
        for i in range(index.__len__()):
            if index[i] == 1:
                svhndata.append(dataset.data[i])
                svhnlabels.append(dataset.labels[i])
        dataset.data = svhndata
        dataset.labels = svhnlabels
        print(dataset.data.__len__())

        # elif(split == 'test'):
        #     index = torch.from_numpy(np.in1d(dataset.test_labels, selected_labels).astype(np.uint8))
        #     dataset.data = dataset.data[index]
        #     dataset.labels = dataset.labels[index]
        # else:
        #     print('wrong split in svhn')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_data_loader_mnist(batch_size, train, new_size=32, num_workers=4, selected_labels=None, selected_scaled_labels=None, selected_labels_scale=0.5):
    transform_list = [transforms.ToTensor(),
                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)
    dataset = datasets.MNIST(root='datasets/mnist', download=True, transform=transform, train=train)
    if train:
        if selected_labels is not None:  # only use data with selected labels
            print('entering selected_labels circle for mnist')
            index = torch.from_numpy(np.in1d(dataset.train_labels, selected_labels).astype(np.uint8))
        else:
            selected_labels = []
            index = torch.from_numpy(np.in1d(dataset.train_labels, selected_labels).astype(np.uint8))
        print(index.__len__())

        if selected_scaled_labels is not None:
            print('entering selected_scaled_labels circle for mnist')
            for scale_label in selected_scaled_labels:
                index1 = torch.from_numpy(np.in1d(dataset.train_labels, scale_label).astype(np.uint8))
                index1non = np.nonzero(index1)
                len_index1non = len(index1non)
                order = np.array(range(len_index1non))
                random.shuffle(order)
                for i in order[int(len_index1non * (int(scale_label) * 0.1)):len_index1non]:
                    index1[index1non[i]] = 0
                index = index + index1
        dataset.train_data = dataset.train_data[index]
        dataset.train_labels = dataset.train_labels[index]
        print(dataset.train_labels.__len__())
        # else:
        #     index = torch.from_numpy(np.in1d(dataset.test_labels, selected_labels).astype(np.uint8))
        #     dataset.test_data = dataset.test_data[index]
        #     dataset.test_labels = dataset.test_labels[index]
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_usps(batch_size, train, new_size=32, num_workers=4, selected_labels=None, selected_scaled_labels=None, selected_labels_scale=0.5):
    transform_list = [transforms.ToTensor(),
                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)
    dataset = usps.USPS(root='datasets/usps', download=False, transform=transform, train=train)

    if train:
        if selected_labels is not None:  # only use data with selected labels
            print('entering selected_table circle *** src:usps')
            index = torch.from_numpy(np.in1d(dataset.targets, selected_labels).astype(np.uint8))

        else:
            selected_labels = []
            index = torch.from_numpy(np.in1d(dataset.targets, selected_labels).astype(np.uint8))
        print(index.__len__())


        if selected_scaled_labels is not None:
            print('entering selected_scaled_labels circle *** src:usps')
            for scale_label in selected_scaled_labels:
                index1 = torch.from_numpy(np.in1d(dataset.targets, scale_label).astype(np.uint8))
                index1non = np.nonzero(index1)
                len_index1non = len(index1non)
                order = np.array(range(len_index1non))
                random.shuffle(order)
                for i in order[int(len_index1non * (1 - int(scale_label) * 0.1)):len_index1non]:
                    index1[index1non[i]] = 0
                index = index + index1

        images = []
        targets = []
        print('entering rebuilding dataset circle *** src:usps')
        for i in range(index.__len__()):
            if index[i] == 1:
                images.append(dataset.images[i])
                targets.append(dataset.targets[i])
        dataset.targets = targets
        dataset.images = images
        print(dataset.targets.__len__())
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader
###reading eval dataset
def get_Evaldata_loader(images, labels, batch_size=32, train=True, num_workers=0):
    dataset = Evaldata(images, labels, transform=None)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

class Evaldata(torchdata.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels = images, labels

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

###reading celebrity dataset
class celeA(torchdata.Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform
        self.names, self.labels = self.get_all_img_names_and_labels()

    def get_all_img_names_and_labels(self):
        """ You should implement this mathod
        list all self.img_dir's images, stored in self.names
        and get each image' label, stored in self.labels
        """
        names = []
        labels = []
        f = open(self.label_file, 'r')
        for line in f.readlines():
            all = line.split()

            imgname = all[0]
            hair = all[1]

            names.append(imgname)
            labels.append(int(hair))

        return names, labels

    def __getitem__(self, index):
        name, label = self.names[index], self.labels[index]
        fpath = os.path.join(self.img_dir, name)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.names)


def get_all_celeAdata_loaders(conf):
    # if 'new_size' in conf:
    #     new_size_a = new_size_b = conf['new_size']
    # else:
    #     new_size_a = conf['new_size_a']
    #     new_size_b = conf['new_size_b']
    # height = 64
    # width = 64
    # A_labels = conf['A_labels'] if 'A_labels' in conf else None
    # B_labels = conf['B_labels'] if 'B_labels' in conf else None

    if 'new_size' in conf:
        new_size = conf['new_size']
    else:
        new_size = 32

    train_loader_a = get_celeAdata_loader_A(conf['batch_size'], True, new_size=new_size, num_workers=4)
    test_loader_a = get_celeAdata_loader_A(conf['batch_size'], False, new_size=new_size, num_workers=4)
    train_loader_b = get_celeAdata_loader_B(conf['batch_size'], True, new_size=new_size, num_workers=4)
    test_loader_b = get_celeAdata_loader_B(conf['batch_size'], False, new_size=new_size, num_workers=4)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b



def get_celeAdata_loader_A(batch_size, train, new_size=32, num_workers=4):
    imgroot = 'datasets/celeA/img_align_celeba'
    if train:
        # split = 'train'
        # save_root = 'data/celeA/celebA_resized32/train/'
        label_file = 'datasets/celeA/Annotation/Anno_boyi/train_male1.txt'
    else:
        # split = 'test'
        # save_root = 'data/celeA/celebA_resized32/test/'
        label_file = 'datasets/celeA/Annotation/Anno_boyi/test_male1.txt'

    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    transform_list = [transforms.Resize([new_size, new_size])] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)

    dataset = celeA(imgroot, label_file, transform)
    # if not os.path.isdir(save_root):
    #     os.mkdir(save_root)
    # if not os.path.isdir(save_root + 'female/'):
    #     os.mkdir(save_root + 'female/')
    # if not os.path.isdir(save_root + 'male/'):
    #     os.mkdir(save_root + 'male/')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_celeAdata_loader_B(batch_size, train, new_size=32, num_workers=4):
    imgroot = 'datasets/celeA/img_align_celeba'
    if train:
        label_file = 'datasets/celeA/Annotation/Anno_boyi/train_female1.txt'
    else:
        label_file = 'datasets/celeA/Annotation/Anno_boyi/test_female1.txt'

    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize([new_size, new_size])] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)

    dataset = celeA(imgroot, label_file, transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True,
                                         num_workers=num_workers)
    return loader



###################################################
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    if n == 3:
        __write_images(image_outputs[0:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
    else:
        __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
        __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % image_directory, all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % image_directory, all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and
               ('loss' in attr or 'grad' in attr or 'acc' in attr or 'lr' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """
    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base, trainer_name):
    def __conversion_core(state_dict_base, trainer_name):
        state_dict = state_dict_base.copy()
        if trainer_name == 'MUNIT':
            for key, value in state_dict_base.items():
                if key.endswith(('enc_content.model.0.norm.running_mean',
                                 'enc_content.model.0.norm.running_var',
                                 'enc_content.model.1.norm.running_mean',
                                 'enc_content.model.1.norm.running_var',
                                 'enc_content.model.2.norm.running_mean',
                                 'enc_content.model.2.norm.running_var',
                                 'enc_content.model.3.model.0.model.1.norm.running_mean',
                                 'enc_content.model.3.model.0.model.1.norm.running_var',
                                 'enc_content.model.3.model.0.model.0.norm.running_mean',
                                 'enc_content.model.3.model.0.model.0.norm.running_var',
                                 'enc_content.model.3.model.1.model.1.norm.running_mean',
                                 'enc_content.model.3.model.1.model.1.norm.running_var',
                                 'enc_content.model.3.model.1.model.0.norm.running_mean',
                                 'enc_content.model.3.model.1.model.0.norm.running_var',
                                 'enc_content.model.3.model.2.model.1.norm.running_mean',
                                 'enc_content.model.3.model.2.model.1.norm.running_var',
                                 'enc_content.model.3.model.2.model.0.norm.running_mean',
                                 'enc_content.model.3.model.2.model.0.norm.running_var',
                                 'enc_content.model.3.model.3.model.1.norm.running_mean',
                                 'enc_content.model.3.model.3.model.1.norm.running_var',
                                 'enc_content.model.3.model.3.model.0.norm.running_mean',
                                 'enc_content.model.3.model.3.model.0.norm.running_var',
                                 )):
                    del state_dict[key]
        else:
            def __conversion_core(state_dict_base):
                state_dict = state_dict_base.copy()
                for key, value in state_dict_base.items():
                    if key.endswith(('enc.model.0.norm.running_mean',
                                     'enc.model.0.norm.running_var',
                                     'enc.model.1.norm.running_mean',
                                     'enc.model.1.norm.running_var',
                                     'enc.model.2.norm.running_mean',
                                     'enc.model.2.norm.running_var',
                                     'enc.model.3.model.0.model.1.norm.running_mean',
                                     'enc.model.3.model.0.model.1.norm.running_var',
                                     'enc.model.3.model.0.model.0.norm.running_mean',
                                     'enc.model.3.model.0.model.0.norm.running_var',
                                     'enc.model.3.model.1.model.1.norm.running_mean',
                                     'enc.model.3.model.1.model.1.norm.running_var',
                                     'enc.model.3.model.1.model.0.norm.running_mean',
                                     'enc.model.3.model.1.model.0.norm.running_var',
                                     'enc.model.3.model.2.model.1.norm.running_mean',
                                     'enc.model.3.model.2.model.1.norm.running_var',
                                     'enc.model.3.model.2.model.0.norm.running_mean',
                                     'enc.model.3.model.2.model.0.norm.running_var',
                                     'enc.model.3.model.3.model.1.norm.running_mean',
                                     'enc.model.3.model.3.model.1.norm.running_var',
                                     'enc.model.3.model.3.model.0.norm.running_mean',
                                     'enc.model.3.model.3.model.0.norm.running_var',

                                     'dec.model.0.model.0.model.1.norm.running_mean',
                                     'dec.model.0.model.0.model.1.norm.running_var',
                                     'dec.model.0.model.0.model.0.norm.running_mean',
                                     'dec.model.0.model.0.model.0.norm.running_var',
                                     'dec.model.0.model.1.model.1.norm.running_mean',
                                     'dec.model.0.model.1.model.1.norm.running_var',
                                     'dec.model.0.model.1.model.0.norm.running_mean',
                                     'dec.model.0.model.1.model.0.norm.running_var',
                                     'dec.model.0.model.2.model.1.norm.running_mean',
                                     'dec.model.0.model.2.model.1.norm.running_var',
                                     'dec.model.0.model.2.model.0.norm.running_mean',
                                     'dec.model.0.model.2.model.0.norm.running_var',
                                     'dec.model.0.model.3.model.1.norm.running_mean',
                                     'dec.model.0.model.3.model.1.norm.running_var',
                                     'dec.model.0.model.3.model.0.norm.running_mean',
                                     'dec.model.0.model.3.model.0.norm.running_var',
                                     )):
                        del state_dict[key]
        return state_dict

    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'], trainer_name)
    state_dict['b'] = __conversion_core(state_dict_base['b'], trainer_name)
    return state_dict