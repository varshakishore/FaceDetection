from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch import autograd
import numpy as np

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

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


#################################################################################
# Discriminator
#################################################################################
class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.new_size = params['new_size']
        self.linear_dim = params['linear_dim']
        self.input_dim = 3
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        assert self.num_scales == 1

        self.cnn, top_dim = self._make_net()
        if params['type'] == 'patch':
            self.dis_head = nn.Conv2d(top_dim, 1, 1, 1, 0)
        elif params['type'] == 'global':
            self.dis_head = nn.Conv2d(top_dim, 1, 4, 1, 0)
        else:
            assert 0, "Unsupported discriminator type: {}".format(params['type'])
        self.cls_head = nn.Conv2d(top_dim, self.linear_dim, 4, 1, 0)

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x, dim

    def forward(self, x, censor_w):
        fea = self.cnn(x)
        #import pdb;pdb.set_trace()
        dis_score = self.dis_head(fea)
        cls_score = self.cls_head(grad_reverse(fea, censor_w))
        return dis_score, cls_score

    def calc_dis_loss(self, input_fake, input_real, labels_fake, censor_w):
        out_fake, pred_fake = self.forward(input_fake.detach(), 0) # do not use b's label
        out_real, _ = self.forward(input_real.detach(), censor_w)

        # Adversarial loss
        if self.gan_type == 'lsgan':
            loss_adv = torch.mean((out_fake - 0)**2) + torch.mean((out_real - 1)**2)
        elif self.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(out_fake.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out_real.data).cuda(), requires_grad=False)
            loss_adv = torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all0) +
                               F.binary_cross_entropy(F.sigmoid(out_real), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        # Classification loss
        loss_cls = F.cross_entropy(pred_fake.mean(3).mean(2), labels_fake) #+ F.cross_entropy(pred_real.mean(3).mean(2), labels_real)

        return loss_adv, loss_cls

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        out0, _ = self.forward(input_fake, 0)
        loss = 0
        if self.gan_type == 'lsgan':
            loss += torch.mean((out0 - 1)**2) # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


    # def calc_gradient_penalty(self, input_real, censor_w=0):
    #     gradient_penalty = 0
    #     input_real = input_real.detach().requires_grad_()
    #     out = self.forward(input_real.detach(), censor_w=censor_w)
    #     out = out[0].mean(3).mean(2)  # average across all patch discriminators
    #     gradients = autograd.grad(outputs=out, inputs=input_real, grad_outputs=torch.ones(out.size()).cuda(),
    #                               create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    #     # import pdb; pdb.set_trace()
    #     for gradient in gradients:  # for gradient w.r.t. each input
    #         if gradient is not None:
    #             gradient = gradient.view(gradient.size(0), -1)
    #             gradient_penalty += (gradient.norm(2, dim=1) ** 2).mean()
    #     return gradient_penalty
    def calc_gradient_penalty(self, input_real, censor_w=0):
        gradient_penalty = 0
        # input_real.requires_grad_()
        # out = self.forward(input_real.detach(), censor_w=censor_w)
        input_real = input_real.detach().requires_grad_()
        out = self.forward(input_real, censor_w=censor_w)
        out = out[0].mean(3).mean(2)  # average across all patch discriminators
        gradients = autograd.grad(outputs=out, inputs=input_real, grad_outputs=torch.ones(out.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
        # import pdb; pdb.set_trace()
        for gradient in gradients:  # for gradient w.r.t. each input
            if gradient is not None:
                gradient = gradient.view(gradient.size(0), -1)
                gradient_penalty += (gradient.norm(2, dim=1) ** 2).mean()
        return gradient_penalty

        # 1-d
        # gradient_penalty = 0
        # out = self.forward(input_real.detach(), 0)[0]
        # out = out.mean(3).mean(2)  # average across all patch discriminators
        # gradients = autograd.grad(outputs=out, inputs=input_real,
        #                           grad_outputs=torch.ones(out.size()).cuda(),
        #                           create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
        # for gradient in gradients:  # for gradient w.r.t. each input
        #     if gradient is not None:
        #         gradient = gradient.view(gradient.size(0), -1)
        #         gradient_penalty += (gradient.norm(2, dim=1) ** 2).mean()
        # return gradient_penalty




class CycleGANGen(nn.Module):
    def __init__(self, params):
        super(CycleGANGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        norm = params['norm']

        self.encode_model = []
        self.encode_model += [Conv2dBlock(3, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.encode_model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.encode_model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.encode_model = nn.Sequential(*self.encode_model)

        self.decode_model = []
        # upsampling blocks
        for i in range(n_downsample):
            self.decode_model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.decode_model += [Conv2dBlock(dim, 3, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.decode_model = nn.Sequential(*self.decode_model)

    def forward(self, x):
        x = self.encode_model(x)
        return self.decode_model(x)



class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, 3, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, 3, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, 3,
                           res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params




##################################################################################
# Encoder and Decoders
##################################################################################


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class DEGen(nn.Module):
    def __init__(self, params):
        super(DEGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm = params['norm']
        new_size = params['new_size']
        linear_dim = params['linear_dim']

        if (params['new_size'] == 128):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DE_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, linear_dim=linear_dim)
        elif (params['new_size'] == 64):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DE_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, linear_dim=linear_dim)
        elif (params['new_size'] == 32):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DE_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ, pad_type=pad_type, linear_dim=linear_dim)

        #
        # if params['fixed_step']:
        #     print('DEGEN fixed')
        #     for p in self.parameters():
        #         p.requires_grad = False
    def forward(self, images):
        # reconstruct an image
        c, z = self.enc(images)
        images_recon = self.dec(c, z)
        return images_recon




class DEMMGen(nn.Module):
    def __init__(self, params):
        super(DEMMGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm = params['norm']
        new_size = params['new_size']
        linear_dim = params['linear_dim']
        DEMM_flag = params['DEMM']


        if (new_size == 128):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DEMM_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ, pad_type=pad_type, DEMM_flag=DEMM_flag)
        elif (new_size == 64):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DEMM_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ, pad_type=pad_type, DEMM_flag=DEMM_flag)
        elif (new_size == 32):
            # encoder
            self.enc = DE_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=linear_dim)
            # decoder
            self.dec = DEMM_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ, pad_type=pad_type, DEMM_flag=DEMM_flag)

    def forward(self, images, c):
        # reconstruct an image
        _, z = self.enc(images)
        images_recon = self.dec(c, z)
        return images_recon


class DEVVGen(nn.Module):
    def __init__(self, params):
        super(DEVVGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm = params['norm']
        new_size = params['new_size']
        linear_dim = params['linear_dim']


        if (new_size == 128):
            # encoder
            self.enc = DEVV_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, new_size=128, linear_dim=linear_dim)
        elif (new_size == 64):
            # encoder
            self.enc = DEVV_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, new_size=64, linear_dim=linear_dim)
        elif (new_size == 32):
            # encoder
            self.enc = DEVV_Encoder(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, linear_dim=linear_dim)

    def forward(self, images):
        # reconstruct an image
        c, z, _ = self.enc(images)
        images_recon = self.dec(c, z)
        return images_recon

class DEVVGen_old(nn.Module):
    def __init__(self, params):
        super(DEVVGen_old, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm = params['norm']
        new_size = params['new_size']
        linear_dim = params['linear_dim']


        if (new_size == 128):
            # encoder
            self.enc = DE_EncoderV(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, new_size=128, linear_dim=linear_dim)
        elif (new_size == 64):
            # encoder
            self.enc = DE_EncoderV(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, new_size=64, linear_dim=linear_dim)
        elif (new_size == 32):
            # encoder
            self.enc = DE_EncoderV(n_downsample, n_res, 3, dim, norm=norm, activ=activ, pad_type=pad_type, new_size=new_size, linear_dim=128)
            # decoder
            self.dec = DEVV_Decoder(n_downsample, n_res, self.enc.output_dim, 3, norm=norm, activ=activ,
                                        pad_type=pad_type, linear_dim=linear_dim)

    def forward(self, images, c):
        # reconstruct an image
        z = self.enc(images)
        images_recon = self.dec(c, z)
        return images_recon



class DE_EncoderM(nn.Module):
    def __init__(self, params):
        super(DE_EncoderM, self).__init__()
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        input_dim = params['input_dim']
        dim = params['dim']
        norm = params['norm']
        activ = params['activ']
        pad_type = params['pad_type']
        self.new_size = params['new_size']
        linear_dim = params['linear_dim']
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

        if(params['DEMM'] == 1):
            if (self.new_size == 64):
                self.c_branch64 = Conv2dBlock(1, 1, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
            if (self.new_size == 128):
                self.c_branch128 = Conv2dBlock(1, 1, 4, 4, 0, norm=norm, activation=activ, pad_type=pad_type)
            self.c_branch = Conv2dBlock(dim, 1, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)
            self.c_vector_branch = nn.Linear(64, linear_dim)
        else:
            if (self.new_size == 64):
                self.c_branch64 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
            if (self.new_size == 128):
                self.c_branch128 = Conv2dBlock(dim, dim, 4, 4, 0, norm=norm, activation=activ, pad_type=pad_type)
            self.c_branch = Conv2dBlock(dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)
            self.c_vector_branch = nn.Linear(64 * dim, linear_dim)

        self.output_dim = dim

    def forward(self, x):
        #import pdb;pdb.set_trace()
        h = self.model(x)
        c = self.c_branch(h)
        if (self.new_size == 64):
            hc = self.c_branch64(c)
            c_vector = self.c_vector_branch(hc.view(hc.size(0), -1))
        if (self.new_size == 128):
            hc = self.c_branch128(c)
            c_vector = self.c_vector_branch(hc.view(hc.size(0), -1))
        else:
            c_vector = self.c_vector_branch(c.view(c.size(0), -1))
        return c, c_vector

class DE_EncoderV(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, new_size=32, linear_dim=10):
        super(DE_EncoderV, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        if (new_size == 64):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        if (new_size == 128):
            self.model += [Conv2dBlock(dim, dim, 4, 4, 0, norm=norm, activation=activ, pad_type=pad_type)]
        #residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

        self.z_branch = nn.Linear(64 * dim, linear_dim) #nn.Linear(16 * dim, 10)
        self.output_dim = linear_dim

    def forward(self, x):
        #import pdb;pdb.set_trace()
        h = self.model(x)
        z = self.z_branch(h.view(h.size(0), -1))
        # b, n = z.shape
        # predict = torch.argmax(z, dim=-1, keepdim=True)
        # z = z.new_zeros((b, n)).scatter_(1, predict, 1)
        return z


class DEMM_Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='adain', activ='relu', pad_type='zero', out_activ='tanh', DEMM_flag=0):
        super(DEMM_Decoder, self).__init__()

        if(DEMM_flag == 1):
            self.conv = Conv2dBlock(dim + 1, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
        else:
            self.conv = Conv2dBlock(dim * 2, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)

        self.model = []
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=out_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        # c embedding

    def forward(self, c, z):
        return self.model(self.conv(torch.cat([c, z], dim=1)))

class DEVV_Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='adain', activ='relu', pad_type='zero', out_activ='tanh', new_size=32, linear_dim=10):
        super(DEVV_Decoder, self).__init__()
        self.new_size = new_size
        self.conv = Conv2dBlock(dim * 2, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
        self.embedding = LinearBlock(linear_dim, dim, norm=norm, activation=activ)
        self.model = []
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type)]
        # if (new_size == 64):
        #     self.model += [nn.Upsample(scale_factor=2), Conv2dBlock(dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        # if (new_size == 128):
        #     self.model += [nn.Upsample(scale_factor=4), Conv2dBlock(dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=out_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        # c embedding

    # def K(self, c, c_noise):
    #     out = 1 - torch.div(c_noise.float(), c.float() + 0.001)
    #     k = F.relu(out)
    #     out = torch.mul(c, k)
    #     return out

    def forward(self, c, z):
        c = self.embedding(c).unsqueeze(-1).unsqueeze(-1)
        z = z.unsqueeze(-1).unsqueeze(-1)
        size = int(self.new_size/4)
        c = c.expand(-1, -1, size, size)
        z = z.expand(-1, -1, size, size)

        return self.model(self.conv(torch.cat([c, z], dim=1)))

class DEVV_Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, new_size=32, linear_dim=10, fixed_step=0):
        super(DEVV_Encoder, self).__init__()
        self.new_size = new_size
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        if (self.new_size == 64):
            self.c_branch64 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        if (self.new_size == 128):
            self.c_branch128 = Conv2dBlock(dim, dim, 4, 4, 0, norm=norm, activation=activ, pad_type=pad_type)
        self.c_branch = nn.Linear(64 * dim, linear_dim) #nn.Linear(16 * dim, 10)

        self.z_branch = nn.Linear(linear_dim, 10)

        self.output_dim = linear_dim

        # # # fixed first step
        # if fixed_step:
        #     print('DE_Encoder Z fixed')
        #     for p in self.parameters():
        #         p.requires_grad = False

    def forward(self, x):
        #import pdb;pdb.set_trace()
        h = self.model(x)
        if (self.new_size == 64):
            hc = self.c_branch64(h)
            c = self.c_branch(hc.view(hc.size(0), -1))
        elif (self.new_size == 128):
            hc = self.c_branch128(h)
            c = self.c_branch(hc.view(hc.size(0), -1))
        else:
            c = self.c_branch(h.view(h.size(0), -1))
        # z = self.z_branch(h.view(h.size(0), -1))
        # z = c[:, self.linear_dim:self.linear_dim + 128]
        # c = c[:, 0:self.linear_dim]
        z = c
        z_c = self.z_branch(grad_reverse(z, 1))
        return z, z_c

class DE_Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, new_size=32, linear_dim=10, fixed_step=0):
        super(DE_Encoder, self).__init__()
        self.new_size = new_size
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        if (self.new_size == 64):
            self.c_branch64 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        if (self.new_size == 128):
            self.c_branch128 = Conv2dBlock(dim, dim, 4, 4, 0, norm=norm, activation=activ, pad_type=pad_type)
        self.c_branch = nn.Linear(64 * dim, linear_dim) #nn.Linear(16 * dim, 10)

        self.z_branch = Conv2dBlock(dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)

        self.output_dim = dim

        # # fixed first step
        if fixed_step:
            print('DE_Encoder Z fixed')
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        #import pdb;pdb.set_trace()
        h = self.model(x)
        if (self.new_size == 64):
            hc = self.c_branch64(h)
            c = self.c_branch(hc.view(hc.size(0), -1))
        elif (self.new_size == 128):
            hc = self.c_branch128(h)
            c = self.c_branch(hc.view(hc.size(0), -1))
        else:
            c = self.c_branch(h.view(h.size(0), -1))
        #c = torch.nn.Softmax(c)
        z = self.z_branch(h)
        return c, z


class DE_Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='adain', activ='relu', pad_type='zero', out_activ='tanh', linear_dim=10):
        super(DE_Decoder, self).__init__()

        self.conv = Conv2dBlock(dim * 2, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
        self.embedding = LinearBlock(linear_dim, dim, norm=norm, activation=activ)
        self.model = []
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=out_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        # c embedding



    def forward(self, c, z):
        c = self.embedding(c).unsqueeze(-1).unsqueeze(-1)
        c = c.expand(-1, -1, z.size(2), z.size(3))
        return self.model(self.conv(torch.cat([c, z], dim=1)))

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero', out_activ='tanh'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=out_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# class InterClassifier(nn.Module):
#     # From Cycada: https://github.com/jhoffman/cycada_release
#     def __init__(self, input_nc=256, ndf=64, norm_layer=nn.BatchNorm2d):
#         super(InterClassifier, self).__init__()
#
#         kw = 4 #3
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 4
#         nf_mult_prev = 1
#         for n in range(1):
#             #nf_mult_prev = nf_mult
#             #nf_mult = min(2 ** n, 2 ** 1)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=8, padding=2),
#                 norm_layer(ndf * nf_mult, affine=True),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         self.before_linear = nn.Sequential(*sequence)
#
#         sequence = [
#             nn.Linear(ndf * nf_mult, 1024),
#             nn.Linear(1024, 10)
#         ]
#
#         self.after_linear = nn.Sequential(*sequence)
#
#
#     def forward(self, x):
#         bs = x.size(0)
#         out = self.after_linear(self.before_linear(x).view(bs, -1))
#         return out

class Classifier(nn.Module):
    # From Cycada: https://github.com/jhoffman/cycada_release
    def __init__(self, linear_dim=10, new_size=32, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()
        kw = 3
        nf_mult = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
            nn.LeakyReLU(0.2, True)
        ]


        inputsize = new_size
        if (inputsize == 64):
            sequence += [nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2), norm_layer(ndf, affine=True),
                         nn.LeakyReLU(0.2, True)]
        elif (inputsize == 128):
            sequence += [nn.Conv2d(ndf, ndf, kernel_size=kw, stride=4), norm_layer(ndf, affine=True),
                         nn.LeakyReLU(0.2, True)]

        for n in range(3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        self.before_linear = nn.Sequential(*sequence)

        sequence = [
            nn.Linear(ndf * nf_mult, 1024),
            nn.Linear(1024, linear_dim)
        ]

        self.after_linear = nn.Sequential(*sequence)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        bs = x.size(0)
        out = self.after_linear(self.before_linear(x).view(bs, -1))

        return out