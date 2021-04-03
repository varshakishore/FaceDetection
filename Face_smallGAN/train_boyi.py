from utils_boyi import get_mnist2svhn_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, \
    set_requires_grad, get_all_celeAdata_loaders, get_data_loader_usps, get_mnist2usps_data_loaders
import argparse
from trainer_boyi import DE_Trainer, DEVV_Trainer, DEVM_Trainer, DEMM_Trainer
from trainer import MUNIT_Trainer, CycleGAN_Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import math
import random
from torchsummary import summary
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist_svhn_baseline_cls1_approach3_lr001.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--trainer', type=str, default='DEVVGAN')
parser.add_argument("--resume", action="store_true")
parser.add_argument('--pretrain', action="store_true", help="pretrain the classifier or vector classifier only")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
#content_encodershare = config['content_encodershare']
config['vgg_model_path'] = opts.output_path
#change the value
if config['dis']['type'] == 'global':
    config['dis']['n_layer'] = int(math.log(config['new_size'], 2) - 2)

print(opts.trainer)
# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
elif opts.trainer == 'CycleGAN':
    trainer = CycleGAN_Trainer(config)
elif opts.trainer == 'DEGAN':
    trainer = DE_Trainer(config)
elif opts.trainer == 'DEVVGAN':
    trainer = DEVV_Trainer(config)
elif opts.trainer == 'DEVMGAN':
    trainer = DEVM_Trainer(config)
elif opts.trainer == 'DEMMGAN':
    trainer = DEMM_Trainer(config)
else:
    sys.exit("Unsupported trainer: {}".format(opts.trainer))
trainer.cuda()

if opts.pretrain:  # prepare to pretrain the classifier
    best_acc_a = 0
    best_acc_b = 0
    config['lr'] = 0.001
else:   # load the pretrained classifier
    classifier_a_name = os.path.join(config['classifier_root'], 'classifier_a_best.pt')
    classifier_b_name = os.path.join(config['classifier_root'], 'classifier_b_best.pt')
    if(config['a2b'] == 1):
        trainer.classifier_a.load_state_dict(torch.load(classifier_a_name))
        trainer.classifier_b.load_state_dict(torch.load(classifier_b_name))
    else:
        trainer.classifier_a.load_state_dict(torch.load(classifier_b_name))
        trainer.classifier_b.load_state_dict(torch.load(classifier_a_name))
    set_requires_grad(trainer.classifier_a, False)
    set_requires_grad(trainer.classifier_b, False)


print(config['datasetinfo'] + ' ' + str(config['a2b']) + ' self_training:' + str(config['self_training']))
if config['preserved']:
    print('Preserved: only change the label we need')

if(config['datasetinfo'] == 'mnist_svhn'):
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_mnist2svhn_data_loaders(config)
elif(config['datasetinfo'] == 'mnist_usps'):
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_mnist2usps_data_loaders(config)
elif(config['datasetinfo'] == 'celebA'):
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_celeAdata_loaders(config)


train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
train_display_label_a = torch.stack([torch.tensor(train_loader_a.dataset[i][1]) for i in range(display_size)]).cuda()
train_display_label_b = torch.stack([torch.tensor(train_loader_b.dataset[i][1])for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
test_display_label_a = torch.stack([torch.tensor(test_loader_a.dataset[i][1]) for i in range(display_size)]).cuda()
test_display_label_b = torch.stack([torch.tensor(test_loader_b.dataset[i][1]) for i in range(display_size)]).cuda()


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder
if not os.path.exists(config['classifier_root']):
    print("Creating directory: {}".format(config['classifier_root']))
    os.makedirs(config['classifier_root'])

# Start training
# if opts.resume:
#     #resume_directory = './outputs/debug_DEMM_mnist2usps_fulllabelT/checkpoints'
#     resume_directory = './outputs/debug_DEMM_mnist2usps_fulllabelT/checkpoints'
#     print('resume from ' + resume_directory)
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
# iterations = 25000
# resume_directory = './outputs/debug_DEVM_mnist2usps_fulllabelT/checkpoints/gen_00025000.pt'
# state_dict = torch.load(resume_directory)
#
# my_dict = trainer.gen_a.dec.model.state_dict()
# state_dict_tmp = {k: v for k, v in state_dict['a'].items() if k in my_dict}
# my_dict.update(state_dict_tmp)
# trainer.gen_a.dec.model.load_state_dict(my_dict)
#
# my_dict = trainer.gen_b.dec.model.state_dict()
# state_dict_tmp = {k: v for k, v in state_dict['a'].items() if k in my_dict}
# my_dict.update(state_dict_tmp)
# trainer.gen_b.dec.model.load_state_dict(my_dict)
#
# my_dict = trainer.gen_a.dec.model.state_dict()
# state_dict_tmp = {k: v for k, v in state_dict['a'].items() if k in my_dict}
# my_dict.update(state_dict_tmp)
# trainer.gen_a.dec.model.load_state_dict(my_dict)
#
# my_dict = trainer.gen_b.enc.state_dict()
# state_dict_tmp = {k: v for k, v in state_dict['a'].items() if k in my_dict}
# my_dict.update(state_dict_tmp)
# trainer.gen_b.enc.load_state_dict(my_dict)

# ending loading model
while True:
    for it, train_data in enumerate(zip(train_loader_a, train_loader_b)):
        data_a, data_b = train_data
        images_a, labels_a = data_a
        images_b, labels_b = data_b
        if config['self_training']:
            index = np.arange(0, len(images_b))
            random.shuffle(index)
            for i in range(0, len(index)):
                images_b = torch.stack([images_b[i] for i in index])
                labels_b = torch.stack([labels_b[i] for i in index])

        images_a, images_b = images_a.cuda(), images_b.cuda()
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()
        trainer.update_learning_rate()

        # pretrain the classifier
        if opts.pretrain:
            trainer.classifier_update(images_a, images_b, labels_a, labels_b)
            # print('classifier iteration')
            # print(iterations+1)
            # print((iterations + 1) % config['log_iter'])
            if (iterations + 1) % config['log_iter'] == 0:
                print("Domain A classification training accuracy: {}".format(trainer.train_acc_a))
                print("Domain B classification training accuracy: {}".format(trainer.train_acc_b))
                write_loss(iterations, trainer, train_writer)
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                test_acc_a_list = []
                test_acc_b_list = []
                for it, test_data in enumerate(zip(test_loader_a, test_loader_b)):  # perform validation
                    test_data_a, test_data_b = test_data
                    test_images_a, test_labels_a = test_data_a
                    test_images_b, test_labels_b = test_data_b
                    test_images_a, test_images_b = test_images_a.cuda(), test_images_b.cuda()
                    test_labels_a, test_labels_b = test_labels_a.cuda(), test_labels_b.cuda()
                    #vutils.save_image(test_images_b, '%s/input_%s.jpg' % (config['output_directory'],))
                    test_acc_a, test_acc_b = trainer.classifier_evaluate(test_images_a, test_images_b, test_labels_a, test_labels_b)
                    test_acc_a_list.append(test_acc_a)
                    test_acc_b_list.append(test_acc_b)
                test_acc_a_final = np.mean(test_acc_a_list)
                test_acc_b_final = np.mean(test_acc_b_list)
                trainer.test_acc_a = torch.Tensor([test_acc_a_final])
                trainer.test_acc_b = torch.Tensor([test_acc_b_final])
                if test_acc_a_final > best_acc_a:
                    best_acc_a = test_acc_a_final
                    best_a = True
                else:
                    best_a = False
                if test_acc_b_final > best_acc_b:
                    best_acc_b = test_acc_b_final
                    best_b = True
                else:
                    best_b = False
                trainer.save_classifiers(config['classifier_root'], iterations, best_a=best_a, best_b=best_b)
                write_loss(iterations, trainer, train_writer)
                print("Domain A classification test accuracy: {}".format(test_acc_a_final))
                print("Domain B classification test accuracy: {}".format(test_acc_b_final))
            # if only train clineassifier
            if opts.pretrain:
                iterations += 1
                if iterations >= max_iter:
                    sys.exit('Finish training')
                continue  # skip the following code

        # train the generators and discriminators
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(images_a, images_b, labels_a, labels_b, config)
            trainer.gen_update(images_a, images_b, labels_a, labels_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b, test_display_label_a, test_display_label_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b, train_display_label_a, train_display_label_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b, train_display_label_a, train_display_label_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)
            #save classifier
            #trainer.save_classifiers(config['classifier_root'], iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
