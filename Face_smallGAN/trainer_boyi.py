from networks_boyi import AdaINGen, CycleGANGen, Classifier, Discriminator, DEGen, DE_EncoderV, DE_EncoderM, DEVVGen, DEMMGen, DEVVGen_old
from utils_boyi import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, set_requires_grad
#from networks import Discriminator
from torch.autograd import Variable
import time
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))


    def onehot(self, y, num_classes=10):
        c_onehot = F.embedding(y, torch.eye(num_classes, device=y.device))
        return c_onehot

    def convert_vector_to_onehot(self, c):
        """
        vector is bxn-dim vector.
        there are n classes and batch size is b.
        """
        b, n = c.shape
        predict = torch.argmax(c, dim=-1, keepdim=True)
        c = c.new_zeros((b, n)).scatter_(1, predict, 1)
        return c

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        #load encoderC
        last_model_name = get_model_list(checkpoint_dir, "encoderC")
        state_dict = torch.load(last_model_name)
        self.encoderC.load_state_dict(state_dict['c'])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        # import pdb;pdb.set_trace()
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        self.encoderC_opt.load_state_dict(state_dict['encoderC'])

        self.classifier_opt.load_state_dict(state_dict['classifier'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.encoderC_scheduler = get_scheduler(self.encoderC_opt, hyperparameters, iterations)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        encoderC_name = os.path.join(snapshot_dir, 'encoderC_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'c': self.encoderC.state_dict()}, encoderC_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        #torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), 'classifier': self.classifier_opt.state_dict()}, opt_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), 'encoderC': self.encoderC_opt.state_dict(), 'classifier': self.classifier_opt.state_dict()}, opt_name)

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.encoderC_scheduler is not None:
            self.encoderC_scheduler.step()
        if self.classifier_scheduler is not None:
            self.classifier_scheduler.step()
        self.gen_lr = self.gen_scheduler.get_lr()[0]
        self.dis_lr = self.dis_scheduler.get_lr()[0]
        self.encoderC_lr = self.encoderC_scheduler.get_lr()[0]
        self.classifier_lr = self.classifier_scheduler.get_lr()[0]


class CTrainer(nn.Module):
    def __init__(self):
        super(CTrainer, self).__init__()

    def resume(self, checkpoint_dir):
        # Load generators
        last_model_name =checkpoint_dir + 'classifier_a2b_.pt'
        state_dict = torch.load(last_model_name)
        self.classifier_a2b.load_state_dict(state_dict)
        print('Resume from iteration %s' % checkpoint_dir)

class Classifier_Trainer(CTrainer):
    def __init__(self, hyperparameters):
        super(Classifier_Trainer, self).__init__()
        lr = hyperparameters['Classifier_lr']
        self.classifier_a2b = Classifier()
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        classifier_a2b_params = list(self.classifier_a2b.parameters())
        self.classifier_a2b_opt = torch.optim.Adam([p for p in classifier_a2b_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_a2b_scheduler = get_scheduler(self.classifier_a2b_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))

    def forward(self, x):
        # self.eval()
        v = self.classifier_a2b(x)
        return v

    def classifier_a2b_update(self, x_a2b, y_a2b):
        self.classifier_a2b.train()
        self.classifier_a2b_opt.zero_grad()
        pred_a2b = self.classifier_a2b(x_a2b)
        _, max_indices_a2b = torch.max(pred_a2b, 1)
        self.train_acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        self.loss_classifier_a2b = F.cross_entropy(pred_a2b, y_a2b)
        self.loss_classifier = self.loss_classifier_a2b
        self.loss_classifier.backward()
        self.classifier_a2b_opt.step()

    def classifier_a2b_evaluate(self, x_a2b, y_a2b):
        self.classifier_a2b.eval()
        with torch.no_grad():
            pred_a2b = self.classifier_a2b(x_a2b)
            _, max_indices_a2b = torch.max(pred_a2b, 1)
            acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        return acc_a2b

    def save_a2b_classifiers(self, snapshot_dir, iterations, best_a2b):
        # Save classifiers
        classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a2b.state_dict(), classifier_a2b_name)
        if best_a2b:
            classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_best.pt')
            torch.save(self.classifier_a2b.state_dict(), classifier_a2b_name)
    def update_learning_rate(self):
        if self.classifier_a2b_scheduler is not None:
            self.classifier_a2b_scheduler.step()

class DE_Trainer(Trainer):
    def __init__(self, hyperparameters):
        super(DE_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.gen_a = DEGen(hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = DEGen(hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['dis'])  # discriminator for domain b
        self.classifier_a = Classifier()
        self.classifier_b = Classifier()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        classifier_params = list(self.classifier_a.parameters()) + list(self.classifier_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))




    def forward(self, x_a, x_b):
        # self.eval()
        c_a, z_a = self.gen_a.enc(x_a)
        c_b, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_a.dec(c_a, z_b)
        x_ba = self.gen_b.dec(c_b, z_a)
        # self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, y_a, y_b, hyperparameters, mmd_lambda=0.002):
        self.classifier_a.eval()
        self.classifier_b.eval()
        self.gen_opt.zero_grad()
        # c_a = self.classifier_a(x_a)
        # c_b = self.classifier_a(x_b)
        # c_a = self.onehotencode(self.classifier_a(x_a))
        # c_b = self.onehotencode(self.classifier_a(x_b))
        c_a, z_a = self.gen_a.enc(x_a)
        c_b, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        #x_ab_confuse = self.gen_b.dec(c_a, z_b)
        x_ba = self.gen_a.dec(c_b, z_b)
        #x_ab = self.gen_b.dec(c_b, z_a)
        # x_ab_prime = self.gen_b.dec(c_b, z_a) #to test not use z_a's info
        # x_ba_prime = self.gen_a.dec(c_a, z_b)

        # cycle reconstruction loss
        if hyperparameters['recon_x_cyc_w'] > 0:
            c_ab, z_ab = self.gen_b.enc(x_ab)
            c_ba, z_ba = self.gen_a.enc(x_ba)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        else:
            self.loss_gen_cycrecon_x_a = 0
            self.loss_gen_cycrecon_x_b = 0

        # image reconstruction loss
        if hyperparameters['recon_x_w'] > 0:
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
            self.loss_gen_recon_x_a = self.recon_criterion(x_aa, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_bb, x_b)
        else:
            self.loss_gen_recon_x_a = 0
            self.loss_gen_recon_x_b = 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # Classifier loss
        #self.loss_mmd = self.mmd_linear(x_a, x_b)
        self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab), y_a)
        #self.loss_gen_cls_a_confuse = F.cross_entropy(self.classifier_a(x_ab_confuse), y_a)
        #self.loss_gen_cls_a_prime = F.cross_entropy(self.classifier_a(x_ab_prime), y_b)
        #self.loss_gen_cls_b = F.cross_entropy(self.classifier_a(x_ba), y_b)

        # print('loss_mmd')
        # print(mmd_lambda * self.loss_mmd)

        # Latent classification loss
        self.loss_gen_cls_ca = F.cross_entropy(c_a, y_a)
        #self.loss_gen_cls_cb = F.cross_entropy(c_b, y_b)

        # # total loss
        # self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
        #                       hyperparameters['gan_w'] * self.loss_gen_adv_b + \
        #                       hyperparameters['cls_w'] * self.loss_gen_cls_a + \
        #                       mmd_lambda * self.loss_mmd + \
        #                       hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
        #                       hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
        #                       hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
        #                       hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
        #                       hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
        #                       hyperparameters['cls_w'] * self.loss_gen_cls_a
        #                       #hyperparameters['cls_w_c'] * self.loss_gen_cls_cb + \
        #                       #hyperparameters['cls_w'] * self.loss_gen_cls_a_prime

        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['cls_w'] * self.loss_gen_cls_a + \
                              hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b \
                              #+ \
                              #hyperparameters['cls_w'] * self.loss_gen_cls_a

        self.loss_gen_total.backward()
        self.gen_opt.step()

        print('loss_gen_adv_a')
        print(self.loss_gen_adv_a)
        print('loss_gen_adv_b')
        print(self.loss_gen_adv_b)
        print('loss_gen_cls_a')
        print(self.loss_gen_cls_a)
        print('loss_gen_cls_ca')
        print(self.loss_gen_cls_ca)
        # print('loss_gen_cls_a_confuse')
        # print(self.loss_gen_cls_a_confuse)
        print('loss_gen_recon_x_a')
        print(self.loss_gen_recon_x_a)
        print('loss_gen_recon_x_b')
        print(self.loss_gen_recon_x_b)
        print('loss_gen_cycrecon_x_a')
        print(self.loss_gen_cycrecon_x_a)
        print('loss_gen_cycrecon_x_b')
        print(self.loss_gen_cycrecon_x_b)
        print('loss_gen_cls_a')
        print(self.loss_gen_cls_a)
        # print('loss_gen_cls_a_prime')
        # print(self.loss_gen_cls_a_prime)


    def sample(self, x_a, x_b):
        # self.eval()
        with torch.no_grad():
            # c_a = self.classifier_a(x_a)
            # c_b = self.classifier_a(x_b)
            # c_a = self.onehotencode(self.classifier_a(x_a))
            # c_b = self.onehotencode(self.classifier_a(x_b))
            c_a, z_a = self.gen_a.enc(x_a)
            c_b, z_b = self.gen_b.enc(x_b)
            x_ab_ori = self.gen_b.dec(c_a, z_a)
            x_ab = self.gen_b.dec(c_a, z_b) #to test not use z_a's info (c_b, z_a)
            x_ba_ori = self.gen_a.dec(c_b, z_b)
            x_ba = self.gen_a.dec(c_a, z_b)
            c_ab, z_ab = self.gen_b.enc(x_ab_ori)
            c_ba, z_ba = self.gen_a.enc(x_ba_ori)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
        # self.train()
        return x_a, x_aa, x_ab_ori, x_ab, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_bab, x_a

    def dis_update(self, x_a, x_b, y_a, y_b, hyperparameters):
        self.dis_opt.zero_grad()
        # c_a = self.classifier_a(x_a)
        # c_b = self.classifier_a(x_b)
        # c_a = self.onehotencode(self.classifier_a(x_a))
        # c_b = self.onehotencode(self.classifier_a(x_b))
        c_a, z_a = self.gen_a.enc(x_a)
        c_b, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)
        #import pdb;pdb.set_trace()
        # D loss
        self.loss_dis_adv_a, _ = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, y_b, hyperparameters['censor_w'])
        self.loss_dis_adv_b, self.loss_dis_cls_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, y_a, hyperparameters['censor_w'])
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + \
        #                       self.loss_dis_cls_a + self.loss_dis_cls_b
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + self.loss_dis_cls_b
        self.loss_dis_total.backward(retain_graph=True)

        if hyperparameters['lambda'] > 0:
            self.loss_grad_penalty = self.dis_a.calc_gradient_penalty(x_a) + self.dis_b.calc_gradient_penalty(x_b)
            self.loss_grad_penalty = hyperparameters['lambda'] * self.loss_grad_penalty
            self.loss_grad_penalty.backward()
        self.dis_opt.step()
        print('loss_dis_adv_b')
        print(self.loss_dis_adv_b)
        print('loss_dis_cls_b')
        print(self.loss_dis_cls_b)

    def classifier_update(self, x_a, x_b, y_a, y_b):
        self.classifier_a.train()
        self.classifier_b.train()
        self.classifier_opt.zero_grad()
        pred_a = self.classifier_a(x_a)
        pred_b = self.classifier_b(x_b)
        _, max_indices_a = torch.max(pred_a, 1)
        _, max_indices_b = torch.max(pred_b, 1)
        self.train_acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
        self.train_acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        self.loss_classifier_a = F.cross_entropy(pred_a, y_a)
        self.loss_classifier_b = F.cross_entropy(pred_b, y_b)
        self.loss_classifier = self.loss_classifier_a + self.loss_classifier_b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_evaluate(self, x_a, x_b, y_a, y_b):
        self.classifier_a.eval()
        self.classifier_b.eval()
        with torch.no_grad():
            pred_a = self.classifier_a(x_a)
            pred_b = self.classifier_b(x_b)
            _, max_indices_a = torch.max(pred_a, 1)
            _, max_indices_b = torch.max(pred_b, 1)
            acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
            acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        return acc_a, acc_b

    def save_classifiers(self, snapshot_dir, iterations, best_a, best_b):
        # Save classifiers
        classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_%08d.pt' % (iterations + 1))
        classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a.state_dict(), classifier_a_name)
        torch.save(self.classifier_b.state_dict(), classifier_b_name)
        if best_a:
            classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a_name)
        if best_b:
            classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_best.pt')
            torch.save(self.classifier_b.state_dict(), classifier_b_name)




class DEMM_Trainer(Trainer):
    def __init__(self, hyperparameters):
        super(DEMM_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.datasetinfo = hyperparameters['datasetinfo']
        self.encoderC = DE_EncoderM(hyperparameters['gen'])
        self.gen_a = DEMMGen(hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = DEMMGen(hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['dis'])  # discriminator for domain b
        if(hyperparameters['new_size'] == 128):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
        elif(hyperparameters['new_size'] == 64):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
        elif(hyperparameters['new_size'] == 32):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'])
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        encoderC_params = list(self.encoderC.parameters())
        classifier_params = list(self.classifier_a.parameters()) + list(self.classifier_b.parameters()) + list(self.classifier_a2b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.encoderC_opt = torch.optim.Adam([p for p in encoderC_params if p.requires_grad],
                                             lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                               lr=lr, betas=(beta1, beta2),
                                               weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.encoderC_scheduler = get_scheduler(self.encoderC_opt, hyperparameters)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))


    def forward(self, x_a, x_b):
        # self.eval()
        c_a, _ = self.encoderC(x_a)
        c_b, _ = self.encoderC(x_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_a.dec(c_a, z_a)
        x_ba = self.gen_b.dec(c_b, z_b)
        # self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, y_a, y_b, hyperparameters, mmd_lambda=0.002):
        self.classifier_a.eval()
        self.classifier_b.eval()
        self.gen_opt.zero_grad()
        c_a, c_a_vector = self.encoderC(x_a)
        c_b, c_b_vector = self.encoderC(x_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab_ori = self.gen_b.dec(c_a, z_a)
        #x_ab_confuse = self.gen_b.dec(c_a, z_b)
        x_ba_ori = self.gen_a.dec(c_b, z_b)
        #x_ab = self.gen_b.dec(c_b, z_a)
        # x_ab_prime = self.gen_b.dec(c_b, z_a)
        # x_ba_prime = self.gen_a.dec(c_a, z_b)

        # cycle reconstruction loss
        if hyperparameters['recon_x_cyc_w'] > 0:
            c_ab, _ = self.encoderC(x_ab_ori)
            c_ba, _ = self.encoderC(x_ba_ori)
            _, z_ab = self.gen_b.enc(x_ab_ori)
            _, z_ba = self.gen_a.enc(x_ba_ori)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        else:
            self.loss_gen_cycrecon_x_a = 0
            self.loss_gen_cycrecon_x_b = 0

        # image reconstruction loss
        if hyperparameters['recon_x_w'] > 0:
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
            self.loss_gen_recon_x_a = self.recon_criterion(x_aa, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_bb, x_b)
        else:
            self.loss_gen_recon_x_a = 0
            self.loss_gen_recon_x_b = 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba_ori)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab_ori)

        # Classifier loss
        self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab_ori), y_a)

        # Latent classification loss
        self.loss_gen_cls_ca = F.cross_entropy(c_a_vector, y_a)

        # # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['cls_w'] * self.loss_gen_cls_a + \
                              hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        if hyperparameters['MoreCinfo']:
            x_ab_cls1 = self.gen_b.dec(c_a, z_b)
            x_ab_cls2 = self.gen_a.dec(c_a, z_b)
            self.loss_gen_cls_aa = F.cross_entropy(self.classifier_a(x_aa), y_a)
            self.loss_gen_cls_ab_cls1 = F.cross_entropy(self.classifier_a(x_ab_cls1), y_a)
            self.loss_gen_cls_ab_cls2 = F.cross_entropy(self.classifier_a(x_ab_cls2), y_a)
            self.loss_gen_total += hyperparameters['cls_w'] * self.loss_gen_cls_aa + \
                                   hyperparameters['cls_w'] * self.loss_gen_cls_ab_cls1 + \
                                   hyperparameters['cls_w'] * self.loss_gen_cls_ab_cls2

        self.loss_gen_total.backward()
        self.gen_opt.step()

        # print('loss_gen_adv_a')
        # print(self.loss_gen_adv_a)
        # print('loss_gen_adv_b')
        # print(self.loss_gen_adv_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # print('loss_gen_cls_ca')
        # print(self.loss_gen_cls_ca)
        # # print('loss_gen_cls_a_confuse')
        # # print(self.loss_gen_cls_a_confuse)
        # print('loss_gen_recon_x_a')
        # print(self.loss_gen_recon_x_a)
        # print('loss_gen_recon_x_b')
        # print(self.loss_gen_recon_x_b)
        # print('loss_gen_cycrecon_x_a')
        # print(self.loss_gen_cycrecon_x_a)
        # print('loss_gen_cycrecon_x_b')
        # print(self.loss_gen_cycrecon_x_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # # print('loss_gen_cls_a_prime')
        # # print(self.loss_gen_cls_a_prime)


    def sample(self, x_a, x_b):
        # self.eval()
        with torch.no_grad():
            c_a, _ = self.encoderC(x_a)
            c_b, _ = self.encoderC(x_b)
            _, z_a = self.gen_a.enc(x_a)
            _, z_b = self.gen_b.enc(x_b)
            x_ab_ori = self.gen_b.dec(c_a, z_a)
            x_ab = self.gen_b.dec(c_b, z_a) #to test not use z_a's info
            x_ba_ori = self.gen_a.dec(c_b, z_b)
            x_ba = self.gen_a.dec(c_a, z_b)
            c_ab, _ = self.encoderC(x_ab_ori)
            c_ba, _ = self.encoderC(x_ba_ori)
            _, z_ab = self.gen_b.enc(x_ab_ori)
            _, z_ba = self.gen_a.enc(x_ba_ori)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
        # self.train()
        if (self.datasetinfo == 'celebA'):
            x_ab_class = self.gen_a.dec(c_b, z_a)
            x_ba_class = self.gen_b.dec(c_a, z_b)
            return x_a, x_aa, x_ab_ori, x_ab, x_ab_class, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_class, x_bab, x_a
        else:
            return x_a, x_aa, x_ab_ori, x_ab, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_bab, x_a

    def dis_update(self, x_a, x_b, y_a, y_b, hyperparameters):
        self.dis_opt.zero_grad()
        c_a, _ = self.encoderC(x_a)
        c_b, _ = self.encoderC(x_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)
        #import pdb;pdb.set_trace()
        # D loss
        self.loss_dis_adv_a, _ = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, y_b, hyperparameters['censor_w'])
        self.loss_dis_adv_b, self.loss_dis_cls_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, y_a, hyperparameters['censor_w'])
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + \
        #                       self.loss_dis_cls_a + self.loss_dis_cls_b
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + self.loss_dis_cls_b
        self.loss_dis_total.backward(retain_graph=True)

        if hyperparameters['lambda'] > 0:
            self.loss_grad_penalty = self.dis_a.calc_gradient_penalty(x_a, hyperparameters['censor_w']) + self.dis_b.calc_gradient_penalty(x_b, hyperparameters['censor_w'])
            self.loss_grad_penalty = hyperparameters['lambda'] * self.loss_grad_penalty
            #import pdb;pdb.set_trace()
            self.loss_grad_penalty.backward(retain_graph=True)
        self.dis_opt.step()
        # print('loss_dis_adv_b')
        # print(self.loss_dis_adv_b)
        # print('loss_dis_adv_a')
        # print(self.loss_dis_adv_a)
        # print('loss_dis_cls_b')
        # print(self.loss_dis_cls_b)

    def classifier_a2b_update(self, x_a2b, y_a2b):
        self.classifier_a2b.train()
        self.classifier_opt.zero_grad()
        pred_a2b = self.classifier_a2b(x_a2b)
        _, max_indices_a2b = torch.max(pred_a2b, 1)
        self.train_acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        self.loss_classifier_a2b = F.cross_entropy(pred_a2b, y_a2b)
        self.loss_classifier = self.loss_classifier_a2b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_a2b_evaluate(self, x_a2b, y_a2b):
        self.classifier_a2b.eval()
        with torch.no_grad():
            pred_a2b = self.classifier_a2b(x_a2b)
            _, max_indices_a2b = torch.max(pred_a2b, 1)
            acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        return acc_a2b

    def save_a2b_classifiers(self, snapshot_dir, iterations, best_a2b):
        # Save classifiers
        classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a2b.state_dict(), classifier_a2b_name)
        if best_a2b:
            classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a2b_name)

    def classifier_update(self, x_a, x_b, y_a, y_b):
        self.classifier_a.train()
        self.classifier_b.train()
        self.classifier_opt.zero_grad()
        pred_a = self.classifier_a(x_a)
        pred_b = self.classifier_b(x_b)
        _, max_indices_a = torch.max(pred_a, 1)
        _, max_indices_b = torch.max(pred_b, 1)
        self.train_acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
        self.train_acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        self.loss_classifier_a = F.cross_entropy(pred_a, y_a)
        self.loss_classifier_b = F.cross_entropy(pred_b, y_b)
        self.loss_classifier = self.loss_classifier_a + self.loss_classifier_b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_evaluate(self, x_a, x_b, y_a, y_b):
        self.classifier_a2b.eval()
        self.classifier_b.eval()
        with torch.no_grad():
            pred_a = self.classifier_a(x_a)
            pred_b = self.classifier_b(x_b)
            _, max_indices_a = torch.max(pred_a, 1)
            _, max_indices_b = torch.max(pred_b, 1)
            acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
            acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        return acc_a, acc_b

    def save_classifiers(self, snapshot_dir, iterations, best_a, best_b):
        # Save classifiers
        classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_%08d.pt' % (iterations + 1))
        classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a.state_dict(), classifier_a_name)
        torch.save(self.classifier_b.state_dict(), classifier_b_name)
        if best_a:
            classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a_name)
        if best_b:
            classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_best.pt')
            torch.save(self.classifier_b.state_dict(), classifier_b_name)

class DEVM_Trainer(Trainer):
    def __init__(self, hyperparameters):
        super(DEVM_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.self_training = hyperparameters['self_training']
        self.datasetinfo = hyperparameters['datasetinfo']
        self.encoderC = DE_EncoderV(hyperparameters['gen']['n_downsample'], hyperparameters['gen']['n_res'], 3,
                                    hyperparameters['gen']['dim'], hyperparameters['gen']['norm'], hyperparameters['gen']['activ'],
                                    hyperparameters['gen']['pad_type'], hyperparameters['gen']['new_size'], hyperparameters['gen']['linear_dim'])
        self.gen_a = DEGen(hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = DEGen(hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['dis'])  # discriminator for domain b
        if(hyperparameters['new_size'] == 128):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
        elif(hyperparameters['new_size'] == 64):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
        elif(hyperparameters['new_size'] == 32):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'])
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        encoderC_params = list(self.encoderC.parameters())
        classifier_params = list(self.classifier_a.parameters()) + list(self.classifier_b.parameters()) + list(self.classifier_a2b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.encoderC_opt = torch.optim.Adam([p for p in encoderC_params if p.requires_grad],
                                             lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                               lr=lr, betas=(beta1, beta2),
                                               weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.encoderC_scheduler = get_scheduler(self.encoderC_opt, hyperparameters)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))


    def forward(self, x_a, x_b):
        # self.eval()
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_a.dec(c_a, z_a)
        x_ba = self.gen_b.dec(c_b, z_b)
        # self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, y_a, y_b, hyperparameters, mmd_lambda=0.002):
        self.classifier_a.eval()
        self.classifier_b.eval()
        self.gen_opt.zero_grad()
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        # Latent classification loss
        self.loss_gen_cls_ca = F.cross_entropy(c_a, y_a)

        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)

        # Classifier loss
        self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab), y_a)


        self.loss_gen_cls_a_prime = 0
        for i in range(10):
            y_a_prime = (y_a + i) % 10
            c_a_prime = self.onehot(y_a_prime)
            x_ab_prime = self.gen_b.dec(c_a_prime, z_a)
            self.loss_gen_cls_a_prime += F.cross_entropy(self.classifier_a(x_ab_prime), y_a_prime)
            self.loss_gen_cls_a_prime += F.cross_entropy(self.encoderC(x_ab_prime), y_a_prime)
        self.loss_gen_cls_a_prime += F.cross_entropy(self.encoderC(x_ab), y_a_prime)


        # cycle reconstruction loss
        if hyperparameters['recon_x_cyc_w'] > 0:
            c_ab = self.encoderC(x_ab)
            c_ba = self.encoderC(x_ba)
            _, z_ab = self.gen_b.enc(x_ab)
            _, z_ba = self.gen_a.enc(x_ba)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        else:
            self.loss_gen_cycrecon_x_a = 0
            self.loss_gen_cycrecon_x_b = 0

        # image reconstruction loss
        x_aa = self.gen_a.dec(c_a, z_a)
        x_bb = self.gen_b.dec(c_b, z_b)
        if hyperparameters['recon_x_w'] > 0:
            self.loss_gen_recon_x_a = self.recon_criterion(x_aa, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_bb, x_b)
        else:
            self.loss_gen_recon_x_a = 0
            self.loss_gen_recon_x_b = 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)


        # x_ab_confuse = self.gen_b.dec(c_a, z_b)
        # self.loss_gen_cls_a_confuse += F.cross_entropy(self.classifier_a(x_ab_confuse), y_a)


        # # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['cls_w'] * self.loss_gen_cls_a + \
                              hyperparameters['cls_w_prime'] * self.loss_gen_cls_a_prime + \
                              hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b



        if hyperparameters['MoreCinfo']:
            x_ab_cls1 = self.gen_b.dec(c_a, z_b)
            self.loss_gen_cls_ab_cls1 = F.cross_entropy(self.classifier_a(x_ab_cls1), y_a)
            self.loss_gen_total += hyperparameters['cls_w'] * self.loss_gen_cls_ab_cls1



        self.loss_gen_total.backward()
        self.gen_opt.step()

        # print('loss_gen_adv_a')
        # print(self.loss_gen_adv_a)
        # print('loss_gen_adv_b')
        # print(self.loss_gen_adv_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # print('loss_gen_cls_ca')
        # print(self.loss_gen_cls_ca)
        # # print('loss_gen_cls_a_confuse')
        # # print(self.loss_gen_cls_a_confuse)
        # print('loss_gen_recon_x_a')
        # print(self.loss_gen_recon_x_a)
        # print('loss_gen_recon_x_b')
        # print(self.loss_gen_recon_x_b)
        # print('loss_gen_cycrecon_x_a')
        # print(self.loss_gen_cycrecon_x_a)
        # print('loss_gen_cycrecon_x_b')
        # print(self.loss_gen_cycrecon_x_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # # print('loss_gen_cls_a_prime')
        # # print(self.loss_gen_cls_a_prime)


    def sample(self, x_a, x_b, y_a, y_b):
        # self.eval()
        with torch.no_grad():
            c_a = self.encoderC(x_a)
            c_b = self.encoderC(x_b)
            c_a = self.convert_vector_to_onehot(c_a)
            c_b = self.convert_vector_to_onehot(c_b)
            _, z_a = self.gen_a.enc(x_a)
            _, z_b = self.gen_b.enc(x_b)
            x_ab_ori = self.gen_b.dec(c_a, z_a)
            x_ab = self.gen_b.dec(c_b, z_a) #to test not use z_a's info
            x_ba_ori = self.gen_a.dec(c_b, z_b)
            x_ba = self.gen_a.dec(c_a, z_b)
            c_ab = self.encoderC(x_ab_ori)
            c_ba = self.encoderC(x_ba_ori)
            _, z_ab = self.gen_b.enc(x_ab_ori)
            _, z_ba = self.gen_a.enc(x_ba_ori)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)

            y_a_prime = (y_a + 1) % 10
            c_a_prime = self.onehot(y_a_prime)
            x_ab_prime = self.gen_b.dec(c_a_prime, z_a)

            y_b_prime = (y_b + 1) % 10
            c_b_prime = self.onehot(y_b_prime)
            x_ba_prime = self.gen_a.dec(c_b_prime, z_b)

        # self.train()
        if (self.datasetinfo == 'celebA'):
            x_ab_class = self.gen_a.dec(c_b, z_a)
            x_ba_class = self.gen_b.dec(c_a, z_b)
            return x_a, x_aa, x_ab_ori, x_ab, x_ab_class, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_class, x_bab, x_a
        else:
            #return x_a, x_aa, x_ab_ori, x_ab, x_ab_prime, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_prime, x_bab, x_a
            return x_a, x_aa, x_ab_ori, x_ab_prime, x_aba, x_ab, x_b, x_b, x_bb, x_ba_ori, x_ba_prime, x_bab, x_ba, x_a

    def dis_update(self, x_a, x_b, y_a, y_b, hyperparameters):
        self.dis_opt.zero_grad()
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        _, z_a = self.gen_a.enc(x_a)
        _, z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)
        #import pdb;pdb.set_trace()
        # D loss
        self.loss_dis_adv_a, self.loss_dis_cls_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, y_b, hyperparameters['censor_w'])
        self.loss_dis_adv_b, self.loss_dis_cls_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, y_a, hyperparameters['censor_w'])

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b\
                              + hyperparameters['gan_w'] * self.loss_dis_cls_b
        if self.self_training:
            self.loss_dis_total += self.loss_dis_cls_a
        self.loss_dis_total.backward(retain_graph=True)

        if hyperparameters['lambda'] > 0:
            self.loss_grad_penalty = self.dis_a.calc_gradient_penalty(x_a, hyperparameters['censor_w']) + self.dis_b.calc_gradient_penalty(x_b, hyperparameters['censor_w'])
            self.loss_grad_penalty = hyperparameters['lambda'] * self.loss_grad_penalty
            #import pdb;pdb.set_trace()
            self.loss_grad_penalty.backward(retain_graph=True)
        self.dis_opt.step()
        # print('loss_dis_adv_b')
        # print(self.loss_dis_adv_b)
        # print('loss_dis_adv_a')
        # print(self.loss_dis_adv_a)
        # print('loss_dis_cls_b')
        # print(self.loss_dis_cls_b)

    def classifier_a2b_update(self, x_a2b, y_a2b):
        self.classifier_a2b.train()
        self.classifier_opt.zero_grad()
        pred_a2b = self.classifier_a2b(x_a2b)
        _, max_indices_a2b = torch.max(pred_a2b, 1)
        self.train_acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        self.loss_classifier_a2b = F.cross_entropy(pred_a2b, y_a2b)
        self.loss_classifier = self.loss_classifier_a2b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_a2b_evaluate(self, x_a2b, y_a2b):
        self.classifier_a2b.eval()
        with torch.no_grad():
            pred_a2b = self.classifier_a2b(x_a2b)
            _, max_indices_a2b = torch.max(pred_a2b, 1)
            acc_a2b = (max_indices_a2b == y_a2b).detach().sum().float() / max_indices_a2b.size()[0]
        return acc_a2b

    def save_a2b_classifiers(self, snapshot_dir, iterations, best_a2b):
        # Save classifiers
        classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a2b.state_dict(), classifier_a2b_name)
        if best_a2b:
            classifier_a2b_name = os.path.join(snapshot_dir, 'classifier_a2b_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a2b_name)

    def classifier_update(self, x_a, x_b, y_a, y_b):
        self.classifier_a.train()
        self.classifier_b.train()
        self.classifier_opt.zero_grad()
        pred_a = self.classifier_a(x_a)
        pred_b = self.classifier_b(x_b)
        _, max_indices_a = torch.max(pred_a, 1)
        _, max_indices_b = torch.max(pred_b, 1)
        self.train_acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
        self.train_acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        self.loss_classifier_a = F.cross_entropy(pred_a, y_a)
        self.loss_classifier_b = F.cross_entropy(pred_b, y_b)
        self.loss_classifier = self.loss_classifier_a + self.loss_classifier_b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_evaluate(self, x_a, x_b, y_a, y_b):
        self.classifier_a2b.eval()
        self.classifier_b.eval()
        with torch.no_grad():
            pred_a = self.classifier_a(x_a)
            pred_b = self.classifier_b(x_b)
            _, max_indices_a = torch.max(pred_a, 1)
            _, max_indices_b = torch.max(pred_b, 1)
            acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
            acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        return acc_a, acc_b

    def save_classifiers(self, snapshot_dir, iterations, best_a, best_b):
        # Save classifiers
        classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_%08d.pt' % (iterations + 1))
        classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a.state_dict(), classifier_a_name)
        torch.save(self.classifier_b.state_dict(), classifier_b_name)
        if best_a:
            classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a_name)
        if best_b:
            classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_best.pt')
            torch.save(self.classifier_b.state_dict(), classifier_b_name)


class DEVV_TrainerT(Trainer):
    def __init__(self, hyperparameters):
        super(DEVV_TrainerT, self).__init__()
        lr = hyperparameters['lr']
        self.datasetinfo = hyperparameters['datasetinfo']
        self.encoderC = DE_EncoderV(hyperparameters['gen']['n_downsample'], hyperparameters['gen']['n_res'], 3,
                                    hyperparameters['gen']['dim'], hyperparameters['gen']['norm'],
                                    hyperparameters['gen']['activ'],
                                    hyperparameters['gen']['pad_type'], hyperparameters['gen']['new_size'])
        self.gen_a = DEVVGen(hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = DEVVGen(hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['dis'])  # discriminator for domain b
        if (hyperparameters['new_size'] == 128):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
        elif (hyperparameters['new_size'] == 64):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
        elif (hyperparameters['new_size'] == 32):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'])

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        classifier_params = list(self.classifier_a.parameters()) + list(self.classifier_b.parameters()) + list(self.classifier_a2b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))

    def forward(self, x_a, x_b):
        # self.eval()
        z_a, _ = self.gen_a.enc(x_a)
        z_b, _ = self.gen_b.enc(x_b)
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        # c_a = self.classifier_a(x_a)
        # c_b = self.classifier_a(x_b)
        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        x_ab = self.gen_a.dec(c_a, z_a)
        x_ba = self.gen_b.dec(c_b, z_b)
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, y_a, y_b, hyperparameters, mmd_lambda=0.002):
        self.classifier_a.eval()
        self.classifier_b.eval()
        self.gen_opt.zero_grad()
        z_a, z_a_c = self.gen_a.enc(x_a)
        z_b, _ = self.gen_b.enc(x_b)
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        # c_a = self.classifier_a(x_a)
        # c_b = self.classifier_a(x_b)
        # Latent classification loss
        self.loss_gen_cls_ca = F.cross_entropy(c_a, y_a)
        # Latent classifcation removal from z
        # self.loss_gen_cls_ca += F.cross_entropy(z_a_c, y_a)


        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)

        self.loss_gen_cls_a_prime = 0

        for i in range(10):
            y_a_prime = (y_a + i) % 10
            c_a_prime = self.onehot(y_a_prime)
            x_ab_prime = self.gen_b.dec(c_a_prime, z_a)
            self.loss_gen_cls_a_prime += F.cross_entropy(self.classifier_a(x_ab_prime), y_a_prime)

        #self.loss_gen_cls_a_prime += F.cross_entropy(self.encoderC(x_ab), y_a)

        # cycle reconstruction loss
        if hyperparameters['recon_x_cyc_w'] > 0:
            z_ab, z_ab_c = self.gen_b.enc(x_ab)
            z_ba, _ = self.gen_a.enc(x_ba)
            c_ab = self.encoderC(x_ab)
            c_ba = self.encoderC(x_ba)
            # c_ab = self.classifier_a(x_ab)
            # c_ba = self.classifier_a(x_ba)
            c_ab = self.convert_vector_to_onehot(c_ab)
            c_ba = self.convert_vector_to_onehot(c_ba)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        else:
            self.loss_gen_cycrecon_x_a = 0
            self.loss_gen_cycrecon_x_b = 0

        # image reconstruction loss
        if hyperparameters['recon_x_w'] > 0:
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
            self.loss_gen_recon_x_a = self.recon_criterion(x_aa, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_bb, x_b)
        else:
            self.loss_gen_recon_x_a = 0
            self.loss_gen_recon_x_b = 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # latent Classifier loss
        self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab), y_a)
        # self.loss_gen_cls_a = F.cross_entropy(c_ab, y_a)
        # Latent classifcation removal from z
        # self.loss_gen_cls_a += F.cross_entropy(z_ab_c, y_a)


        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['cls_w'] * self.loss_gen_cls_a + \
                              hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
                              hyperparameters['cls_w_prime'] * self.loss_gen_cls_a_prime +\
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b


        self.loss_gen_total.backward()
        self.gen_opt.step()


    def sample(self, x_a, x_b, y_a, y_b):
        # self.eval()
        with torch.no_grad():
            z_a, _ = self.gen_a.enc(x_a)
            z_b, _ = self.gen_b.enc(x_b)
            c_a = self.encoderC(x_a)
            c_b = self.encoderC(x_b)
            # c_a = self.classifier_a(x_a)
            # c_b = self.classifier_a(x_b)
            c_a = self.convert_vector_to_onehot(c_a)
            c_b = self.convert_vector_to_onehot(c_b)
            x_ab_ori = self.gen_b.dec(c_a, z_a)
            x_ab = self.gen_b.dec(c_b, z_a) #to test not use z_a's info
            x_ba_ori = self.gen_a.dec(c_b, z_b)
            x_ba = self.gen_a.dec(c_a, z_b)
            z_ab, _ = self.gen_b.enc(x_ab_ori)
            z_ba, _ = self.gen_a.enc(x_ba_ori)
            c_ab = self.encoderC(x_ab)
            c_ba = self.encoderC(x_ba)
            # c_ab = self.classifier_a(x_ab)
            # c_ba = self.classifier_a(x_ba)
            c_ab = self.convert_vector_to_onehot(c_ab)
            c_ba = self.convert_vector_to_onehot(c_ba)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)

            y_a_prime = (y_a + 1) % 10
            c_a_prime = self.onehot(y_a_prime)
            x_ab_prime = self.gen_b.dec(c_a_prime, z_a)

            y_b_prime = (y_b + 1) % 10
            c_b_prime = self.onehot(y_b_prime)
            x_ba_prime = self.gen_a.dec(c_b_prime, z_b)

        # self.train()
        if (self.datasetinfo == 'celebA'):
            x_ab_class = self.gen_a.dec(c_b, z_a)
            x_ba_class = self.gen_b.dec(c_a, z_b)
            return x_a, x_aa, x_ab_ori, x_ab, x_ab_class, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_class, x_bab, x_a
        else:
            #return x_a, x_aa, x_ab_ori, x_ab, x_ab_prime, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_prime, x_bab, x_a
            return x_a, x_aa, x_ab_ori, x_ab_prime, x_aba, x_ab, x_b, x_b, x_bb, x_ba_ori, x_ba_prime, x_bab, x_ba, x_a

    def dis_update(self, x_a, x_b, y_a, y_b, hyperparameters):
        self.dis_opt.zero_grad()
        z_a, _ = self.gen_a.enc(x_a)
        z_b, _ = self.gen_b.enc(x_b)
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        # c_a = self.classifier_a(x_a)
        # c_b = self.classifier_a(x_b)
        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)
        #import pdb;pdb.set_trace()
        # D loss
        self.loss_dis_adv_a, _ = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, y_b, hyperparameters['censor_w'])
        self.loss_dis_adv_b, self.loss_dis_cls_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, y_a, hyperparameters['censor_w'])
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + \
        #                       self.loss_dis_cls_a + self.loss_dis_cls_b
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + hyperparameters['gan_w'] * self.loss_dis_cls_b
        self.loss_dis_total.backward(retain_graph=True)

        if hyperparameters['lambda'] > 0:
            self.loss_grad_penalty = self.dis_a.calc_gradient_penalty(x_a) + self.dis_b.calc_gradient_penalty(x_b)
            self.loss_grad_penalty = hyperparameters['lambda'] * self.loss_grad_penalty
            self.loss_grad_penalty.backward()
        self.dis_opt.step()

    def classifier_update(self, x_a, x_b, y_a, y_b):
        self.classifier_a.train()
        self.classifier_b.train()
        self.classifier_opt.zero_grad()
        pred_a = self.classifier_a(x_a)
        pred_b = self.classifier_b(x_b)
        _, max_indices_a = torch.max(pred_a, 1)
        _, max_indices_b = torch.max(pred_b, 1)
        self.train_acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
        self.train_acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        self.loss_classifier_a = F.cross_entropy(pred_a, y_a)
        self.loss_classifier_b = F.cross_entropy(pred_b, y_b)
        self.loss_classifier = self.loss_classifier_a + self.loss_classifier_b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_evaluate(self, x_a, x_b, y_a, y_b):
        self.classifier_a.eval()
        self.classifier_b.eval()
        with torch.no_grad():
            pred_a = self.classifier_a(x_a)
            pred_b = self.classifier_b(x_b)
            _, max_indices_a = torch.max(pred_a, 1)
            _, max_indices_b = torch.max(pred_b, 1)
            acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
            acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        return acc_a, acc_b

    def save_classifiers(self, snapshot_dir, iterations, best_a, best_b):
        # Save classifiers
        classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_%08d.pt' % (iterations + 1))
        classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a.state_dict(), classifier_a_name)
        torch.save(self.classifier_b.state_dict(), classifier_b_name)
        if best_a:
            classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a_name)
        if best_b:
            classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_best.pt')
            torch.save(self.classifier_b.state_dict(), classifier_b_name)

class DEVV_Trainer(Trainer):
    def __init__(self, hyperparameters):
        super(DEVV_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.datasetinfo = hyperparameters['datasetinfo']
        self.encoderC = DE_EncoderV(hyperparameters['gen']['n_downsample'], hyperparameters['gen']['n_res'], 3,
                                    hyperparameters['gen']['dim'], hyperparameters['gen']['norm'], hyperparameters['gen']['activ'],
                                    hyperparameters['gen']['pad_type'], hyperparameters['gen']['new_size'])
        self.gen_a = DEVVGen_old(hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = DEVVGen_old(hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = Discriminator(hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = Discriminator(hyperparameters['dis'])  # discriminator for domain b
        if (hyperparameters['new_size'] == 128):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=128)
        elif (hyperparameters['new_size'] == 64):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'], new_size=64)
        elif (hyperparameters['new_size'] == 32):
            self.classifier_a = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_b = Classifier(linear_dim=hyperparameters['linear_dim'])
            self.classifier_a2b = Classifier(linear_dim=hyperparameters['linear_dim'])

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        encoderC_params = list(self.encoderC.parameters())
        classifier_params = list(self.classifier_a.parameters()) + list(self.classifier_b.parameters())+ list(self.classifier_a2b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.encoderC_opt = torch.optim.Adam([p for p in encoderC_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.encoderC_scheduler = get_scheduler(self.encoderC_opt, hyperparameters)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))




    def forward(self, x_a, x_b):
        # self.eval()
        c_a_ori = self.encoderC(x_a)
        c_b_ori = self.encoderC(x_b)
        c_a = self.convert_vector_to_onehot(c_a_ori)
        c_b = self.convert_vector_to_onehot(c_b_ori)
        z_a = self.gen_a.enc(x_a)
        z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_a.dec(c_a, z_a)
        x_ba = self.gen_b.dec(c_b, z_b)
        # self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, y_a, y_b, hyperparameters, mmd_lambda=0.002):
        self.classifier_a.eval()
        self.classifier_b.eval()
        self.gen_opt.zero_grad()
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)

        # Latent classification loss
        self.loss_gen_cls_ca = F.cross_entropy(c_a, y_a)

        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        z_a = self.gen_a.enc(x_a)
        z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)


        # self.loss_gen_cls_a_prime = 0
        # for i in range(10):
        #     y_a_prime = (y_a + i) % 10
        #     c_a_prime = self.onehot(y_a_prime)
        #     x_ab_prime = self.gen_b.dec(c_a_prime, z_a)
        #     self.loss_gen_cls_a_prime += F.cross_entropy(self.encoderC(x_ab_prime), y_a_prime)
        #
        # self.loss_gen_cls_a_prime += F.cross_entropy(self.encoderC(x_ab), y_a)


        # cycle reconstruction loss
        if hyperparameters['recon_x_cyc_w'] > 0:
            c_ab = self.encoderC(x_ab)
            c_ba = self.encoderC(x_ba)
            c_ab = self.convert_vector_to_onehot(c_ab)
            c_ba = self.convert_vector_to_onehot(c_ba)
            z_ab = self.gen_b.enc(x_ab)
            z_ba = self.gen_a.enc(x_ba)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        else:
            self.loss_gen_cycrecon_x_a = 0
            self.loss_gen_cycrecon_x_b = 0

        # image reconstruction loss
        if hyperparameters['recon_x_w'] > 0:
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)
            self.loss_gen_recon_x_a = self.recon_criterion(x_aa, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_bb, x_b)
        else:
            self.loss_gen_recon_x_a = 0
            self.loss_gen_recon_x_b = 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # Classifier loss
        #self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab), y_a)
        self.loss_gen_cls_a = F.cross_entropy(self.classifier_a(x_ab), y_a)


        # x_ab_confuse = self.gen_b.dec(c_a, z_b)
        # self.loss_gen_cls_a_confuse += F.cross_entropy(self.classifier_a(x_ab_confuse), y_a)






        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['cls_w'] * self.loss_gen_cls_a + \
                              hyperparameters['cls_w_c'] * self.loss_gen_cls_ca + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b


        self.loss_gen_total.backward()
        self.gen_opt.step()

        # print('loss_gen_adv_a')
        # print(self.loss_gen_adv_a)
        # print('loss_gen_adv_b')
        # print(self.loss_gen_adv_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # print('loss_gen_cls_ca')
        # print(self.loss_gen_cls_ca)
        # # print('loss_gen_cls_a_confuse')
        # # print(self.loss_gen_cls_a_confuse)
        # print('loss_gen_recon_x_a')
        # print(self.loss_gen_recon_x_a)
        # print('loss_gen_recon_x_b')
        # print(self.loss_gen_recon_x_b)
        # print('loss_gen_cycrecon_x_a')
        # print(self.loss_gen_cycrecon_x_a)
        # print('loss_gen_cycrecon_x_b')
        # print(self.loss_gen_cycrecon_x_b)
        # print('loss_gen_cls_a')
        # print(self.loss_gen_cls_a)
        # # print('loss_gen_cls_a_prime')
        # # print(self.loss_gen_cls_a_prime)


    def sample(self, x_a, x_b, y_a, y_b):
        # self.eval()
        with torch.no_grad():
            c_a = self.encoderC(x_a)
            c_b = self.encoderC(x_b)
            c_a = self.convert_vector_to_onehot(c_a)
            c_b = self.convert_vector_to_onehot(c_b)
            z_a = self.gen_a.enc(x_a)
            z_b = self.gen_b.enc(x_b)
            x_ab_ori = self.gen_b.dec(c_a, z_a)
            x_ab = self.gen_b.dec(c_b, z_a) #to test not use z_a's info
            x_ba_ori = self.gen_a.dec(c_b, z_b)
            x_ba = self.gen_a.dec(c_a, z_b)
            c_ab = self.encoderC(x_ab_ori)
            c_ba = self.encoderC(x_ba_ori)
            c_ab = self.convert_vector_to_onehot(c_ab)
            c_ba = self.convert_vector_to_onehot(c_ba)
            z_ab = self.gen_b.enc(x_ab_ori)
            z_ba = self.gen_a.enc(x_ba_ori)
            x_aba = self.gen_a.dec(c_ab, z_ab)
            x_bab = self.gen_b.dec(c_ba, z_ba)
            x_aa = self.gen_a.dec(c_a, z_a)
            x_bb = self.gen_b.dec(c_b, z_b)

            y_a_prime = (y_a + 1) % 10
            c_a_prime = self.onehot(y_a_prime)
            x_ab_prime = self.gen_b.dec(c_a_prime, z_a)

            y_b_prime = (y_b + 1) % 10
            c_b_prime = self.onehot(y_b_prime)
            x_ba_prime = self.gen_a.dec(c_b_prime, z_b)

        # self.train()
        if (self.datasetinfo == 'celebA'):
            x_ab_class = self.gen_a.dec(c_b, z_a)
            x_ba_class = self.gen_b.dec(c_a, z_b)
            return x_a, x_aa, x_ab_ori, x_ab, x_ab_class, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_class, x_bab, x_a
        else:
            #return x_a, x_aa, x_ab_ori, x_ab, x_ab_prime, x_aba, x_b, x_b, x_bb, x_ba_ori, x_ba, x_ba_prime, x_bab, x_a
            return x_a, x_aa, x_ab_ori, x_ab_prime, x_aba, x_ab, x_b, x_b, x_bb, x_ba_ori, x_ba_prime, x_bab, x_ba, x_a

    def dis_update(self, x_a, x_b, y_a, y_b, hyperparameters):
        self.dis_opt.zero_grad()
        c_a = self.encoderC(x_a)
        c_b = self.encoderC(x_b)
        c_a = self.convert_vector_to_onehot(c_a)
        c_b = self.convert_vector_to_onehot(c_b)
        z_a = self.gen_a.enc(x_a)
        z_b = self.gen_b.enc(x_b)
        x_ab = self.gen_b.dec(c_a, z_a)
        x_ba = self.gen_a.dec(c_b, z_b)
        #import pdb;pdb.set_trace()
        # D loss
        self.loss_dis_adv_a, _ = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, y_b, hyperparameters['censor_w'])
        self.loss_dis_adv_b, self.loss_dis_cls_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, y_a, hyperparameters['censor_w'])
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + \
        #                       self.loss_dis_cls_a + self.loss_dis_cls_b
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_adv_a + hyperparameters['gan_w'] * self.loss_dis_adv_b + hyperparameters['gan_w'] * self.loss_dis_cls_b
        self.loss_dis_total.backward(retain_graph=True)

        if hyperparameters['lambda'] > 0:
            self.loss_grad_penalty = self.dis_a.calc_gradient_penalty(x_a) + self.dis_b.calc_gradient_penalty(x_b)
            self.loss_grad_penalty = hyperparameters['lambda'] * self.loss_grad_penalty
            self.loss_grad_penalty.backward()
        self.dis_opt.step()
        # print('loss_dis_adv_b')
        # print(self.loss_dis_adv_b)
        # print('loss_dis_adv_a')
        # print(self.loss_dis_adv_a)
        # print('loss_dis_cls_b')
        # print(self.loss_dis_cls_b)

    def classifier_update(self, x_a, x_b, y_a, y_b):
        self.classifier_a.train()
        self.classifier_b.train()
        self.classifier_opt.zero_grad()
        pred_a = self.classifier_a(x_a)
        pred_b = self.classifier_b(x_b)
        _, max_indices_a = torch.max(pred_a, 1)
        _, max_indices_b = torch.max(pred_b, 1)
        self.train_acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
        self.train_acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        self.loss_classifier_a = F.cross_entropy(pred_a, y_a)
        self.loss_classifier_b = F.cross_entropy(pred_b, y_b)
        self.loss_classifier = self.loss_classifier_a + self.loss_classifier_b
        self.loss_classifier.backward()
        self.classifier_opt.step()

    def classifier_evaluate(self, x_a, x_b, y_a, y_b):
        self.classifier_a.eval()
        self.classifier_b.eval()
        with torch.no_grad():
            pred_a = self.classifier_a(x_a)
            pred_b = self.classifier_b(x_b)
            _, max_indices_a = torch.max(pred_a, 1)
            _, max_indices_b = torch.max(pred_b, 1)
            acc_a = (max_indices_a == y_a).detach().sum().float() / max_indices_a.size()[0]
            acc_b = (max_indices_b == y_b).detach().sum().float() / max_indices_b.size()[0]
        return acc_a, acc_b

    def save_classifiers(self, snapshot_dir, iterations, best_a, best_b):
        # Save classifiers
        classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_%08d.pt' % (iterations + 1))
        classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_%08d.pt' % (iterations + 1))
        torch.save(self.classifier_a.state_dict(), classifier_a_name)
        torch.save(self.classifier_b.state_dict(), classifier_b_name)
        if best_a:
            classifier_a_name = os.path.join(snapshot_dir, 'classifier_a_best.pt')
            torch.save(self.classifier_a.state_dict(), classifier_a_name)
        if best_b:
            classifier_b_name = os.path.join(snapshot_dir, 'classifier_b_best.pt')
            torch.save(self.classifier_b.state_dict(), classifier_b_name)
class GradientReversal(torch.autograd.Function):

    def __init__(self, scale_):
        super(GradientReversal, self).__init__()
        self.scale = scale_

    def forward(self, inp):
        return inp.clone()

    def backward(self, grad_out):
        return -self.scale * grad_out.clone()


def grad_reverse(x, lambd):
    return GradientReversal(lambd)(x)
