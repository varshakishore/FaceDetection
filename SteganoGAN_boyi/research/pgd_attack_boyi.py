import numpy as np
from numpy.core.fromnumeric import squeeze 
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import sys
sys.path.append("..")
import torch
import torchvision
from torch.optim import LBFGS
import torch.nn.functional as F
from tqdm import tqdm, trange
import seaborn as sns
import os

def shuffle_params(m):
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())
        
        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))
    if type(m)==nn.BatchNorm2d:
        if "track_running_stats" in m.__dict__:
            m.track_running_stats=False
            
            
class normLayer(nn.Module):
    def __init__(self):
        super(normLayer, self).__init__()
    def forward(self, x):
        b,c,h,w = x.shape
        assert b == 1
        mean = x.view(c, -1).mean(-1)
        std = x.view(c, -1).std(-1)
        x = x - mean.reshape([1, c, 1, 1])
        x = x / (std + 1e-7).reshape([1,c,1,1])
        return x
    
class BasicDecoder(nn.Module):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        modules = []

        modules.append(self._conv2d(3, self.hidden_size))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(normLayer() if self.yan_norm else nn.BatchNorm2d(self.hidden_size))

        for i in range(self.layers-1):
            modules.append(self._conv2d(self.hidden_size, self.hidden_size))
            modules.append(nn.LeakyReLU(inplace=True))
            modules.append(normLayer() if self.yan_norm else nn.BatchNorm2d(self.hidden_size))

        modules.append(self._conv2d(self.hidden_size, self.data_depth))

        self.layers = nn.Sequential(*modules)

        return [self.layers]    

    def __init__(self, data_depth, hidden_size, layers = 3, yan_norm=False):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.yan_norm = yan_norm
        self.layers = layers

        self._models = self._build_models()

    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

def main():
    num_bits = 4
    yan_norm = False
    # models
    # image = "/home/vk352/FaceDetection/datasets/sample/obama2.jpg"
    # image = imread(image, pilmode='RGB') 
    # image.shape

    # image = "/home/vk352/FaceDetection/datasets/data512x512/00001.jpg"
    dir = "/home/vk352/FaceDetection/datasets/data512x512/"
    num_imgs = 25
    loss_mode = "log"
    hinge = 0.3
    steps = 1000
    eps = 0.2
    max_iter = 20
    alpha = 0.5

    # num_imgs = 1
    # steps = 1
    # max_iter = 1

    # [N, 0.09, ]
    # conv_layers_list = [3, 5, 7, 9] 
    # hidden_size_list = [16, 32, 64, 128]
    conv_layers_list = [1, 2, 3, 5, 7] 
    # hidden_size_list = [16, 32, 64, 128] #, 
    hidden_size_list = [256]
    results = np.ones((len(conv_layers_list), len(hidden_size_list)))
    for layer_size_i in range(len(conv_layers_list)):
        for hidden_size_i in range(len(hidden_size_list)):
            sum_final_err = []
            for img_i in range(num_imgs):
                # seed = random.randrange(sys.maxsize) % (2**32 - 1)
                np.random.seed(img_i)
                model = BasicDecoder(num_bits, hidden_size=hidden_size_list[hidden_size_i], layers=conv_layers_list[layer_size_i], yan_norm=yan_norm)
                model.apply(shuffle_params)
                model.to('cuda')

                filename = os.listdir(dir)[img_i]
                image = "{}/{}".format(dir, filename)
                image = imread(image, pilmode='RGB') / 255.0
                image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
                image = image.to('cuda')
                out = model(image)

                target = torch.bernoulli(torch.empty(out.shape).uniform_(0, 1)).to(out.device)
                # target = torch.empty(out.shape).random_(256).to(out.device)
                target.shape

                criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
                criterion1 = torch.nn.L1Loss(reduction='sum')
                criterion2 = torch.nn.MSELoss(reduction='sum')

                def get_loss(outputs, target, loss_mode):
                    if loss_mode == "BCE":
                        loss = criterion(outputs, target)
                    elif loss_mode == "log":
                        loss = -(target * 2 - 1) * outputs
                        loss = torch.nn.functional.softplus(loss)  # log(1+exp(x))
                        loss = torch.sum(loss)
                    elif loss_mode == "hingelog":
                        loss = -(target * 2 - 1) * outputs
                        loss = torch.nn.functional.softplus(loss)  # log(1+exp(x))
                        loss = torch.max(loss-hinge, torch.zeros(target.shape).to(target.device))
                        loss = torch.sum(loss)
                    elif loss_mode == "L1":
                        outputs = F.sigmoid(outputs) * 255
                        loss = criterion1(outputs, target)
                    elif loss_mode == "L2":
                        outputs = F.sigmoid(outputs) * 255
                        loss = criterion2(outputs, target)
                    return loss

                final_err = 0
                adv_image = image.clone().detach()
                


                adv_image = image.clone().detach()
                print("alpha:", alpha)
                error = []

                for i in trange(steps // max_iter):
                    adv_image.requires_grad = True
                    optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)

                    def closure():
                        outputs = model(adv_image)
                        loss = get_loss(outputs, target, loss_mode)

                        optimizer.zero_grad()
                        loss.backward()
                        return loss

                    optimizer.step(closure)
                    delta = torch.clamp(adv_image - image, min=-eps, max=eps)
                    adv_image = torch.clamp(image + delta, min=0, max=1)
                    adv_image = torch.clamp(adv_image*255, 0, 255).int().float()/255.
                    adv_image = adv_image.detach()

                    if loss_mode in ["L1", "L2"]:
                        err = len(torch.nonzero(torch.abs(F.sigmoid(model(adv_image)).float().view(-1)*255-target.view(-1)) > 128)) / target.numel()
                    else:
                        err = len(torch.nonzero((model(adv_image)>0).float().view(-1) != target.view(-1))) / target.numel()
                    #print("error", err)
                    error.append(err)

                final_err = error[-1]
                sum_final_err.append(final_err) 

            # avg_final_err = sum_final_err / float(num_imgs)
            avg_final_err = np.mean(sum_final_err)
            var_final_err = np.std(sum_final_err)
            results[layer_size_i, hidden_size_i] = avg_final_err
            line = "num_bits: {}, conv layer: {}, hidden size: {} ==>  avg_final_err: {}, var_final_err: {}".format(num_bits, conv_layers_list[layer_size_i], hidden_size_list[hidden_size_i], avg_final_err, var_final_err)
            print(line)
            with open('heatmap_scores_4bits.txt','a+') as f: 
                f.writelines('{}\n'.format(line))


    # ax = sns.heatmap(results, xticklabels=conv_layers_list, yticklabels=hidden_size_list, annot=np.round(results*100, 2))
    # plt.xlabel("Depth (num conv layer blocks)")
    # plt.ylabel("Width (num hidden units)")
    # plt.title("Percentage error bits")
    # plt.savefig('save/save_heatmap.png', format='png')
    import pdb; pdb.set_trace()

# result:
# conv layer: 3, hidden size: 16 ==>  avg_final_err: 0.14387207031250002
# conv layer: 3, hidden size: 32 ==>  avg_final_err: 0.07867925008138019
# conv layer: 3, hidden size: 64 ==>  avg_final_err: 0.0747235107421875
# conv layer: 3, hidden size: 128 ==>  avg_final_err: 0.0404925537109375
# conv layer: 3, hidden size: 256 ==>  avg_final_err: 0.04189656575520834
# conv layer: 5, hidden size: 16 ==>  avg_final_err: 0.24352940877278648
# conv layer: 5, hidden size: 32 ==>  avg_final_err: 0.1788860575358073
# conv layer: 5, hidden size: 64 ==>  avg_final_err: 0.14069076538085937
# conv layer: 5, hidden size: 128 ==>  avg_final_err: 0.08699788411458333
# conv layer: 5, hidden size: 256 ==>  avg_final_err: 0.0964567565917969
# conv layer: 7, hidden size: 16 ==>  avg_final_err: 0.2915922546386719
# conv layer: 7, hidden size: 32 ==>  avg_final_err: 0.23537124633789067
# conv layer: 7, hidden size: 64 ==>  avg_final_err: 0.18433888753255212
# conv layer: 7, hidden size: 128 ==>  avg_final_err: 0.15640452067057292
# conv layer: 9, hidden size: 16 ==>  avg_final_err: 0.332468770345052
# conv layer: 9, hidden size: 32 ==>  avg_final_err: 0.2913495381673177
# conv layer: 9, hidden size: 64 ==>  avg_final_err: 0.2414177958170573
# conv layer: 9, hidden size: 128 ==>  avg_final_err: 0.22460835774739582
# conv layer: 13, hidden size: 16 ==>  avg_final_err: 0.3819438680013021
# conv layer: 13, hidden size: 32 ==>  avg_final_err: 0.35014414469401045

# [3, 5, 7, 9, 13]
# [16, 32, 64, 128, 256]
# array([[0.15024511, 0.07453934, 0.06468816, 0.04020899],
#        [0.2469607 , 0.17193115, 0.11621175, 0.09156957],
#        [0.29918493, 0.23740952, 0.18780802, 0.15640452],
#        [0.33246877, 0.29134954, 0.2414178 , 0.22460836],
#        [0.38194387, 0.35014414, 0.31359314, 0.29194799]])
def draw_heatmap():
    # conv_layers_list = [1, 2, 3, 5, 7, 9] 
    # hidden_size_list = [16, 32, 64, 128, 256]
    # results = np.array([[0.14515950520833332, 0.12776102701822917, 0.08521291097005207, 0.08729100545247395, 0.0749919637044271], \
    #                     [0.15058359781901046, 0.06995997111002604, 0.06485824584960936, 0.0400543212890625, 0.027517191569010415], \
    #                     [0.14387207031250002, 0.07867925008138019, 0.0747235107421875, 0.0404925537109375, 0.04189656575520834], \
    #                     [0.24352940877278648, 0.1788860575358073, 0.140690765380859371, 0.08699788411458333, 0.0964567565917969], \
    #                     [0.2915922546386719, 0.23537124633789067, 0.18433888753255212, 0.15640452067057292, 1], \
    #                     [0.33246877, 0.29134954, 0.2414178 , 0.22460836, 1]]).transpose((1,0))

    conv_layers_list = [1, 2, 3, 5, 7] 
    hidden_size_list = [16, 32, 64, 128, 256]
    results = np.array([[0.24923934936523437, 0.2132598876953125, 0.1782093811035156, 0.16701889038085938, 0.14988876342773438], \
                        [0.24275543212890624, 0.16023391723632813, 0.15170101165771485, 0.10872993469238282, 0.10347953796386719], \
                        [0.23978134155273437, 0.18178722381591797, 0.13847164154052735, 0.11226776123046875, 0.09113090515136718], \
                        [0.3101495361328125, 0.2325945281982422, 0.2000906753540039, 0.15941326141357423, 0.14448585510253906], \
                        [0.3373714447021484, 0.28933158874511716, 0.24509742736816406 , 1, 1]]).transpose((1,0))

    ax = sns.heatmap(results, xticklabels=conv_layers_list, yticklabels=hidden_size_list, annot=np.round(results*100, 2))
    plt.xlabel("Depth (num conv layer blocks)")
    plt.ylabel("Width (num hidden units)")
    plt.title("Percentage error bits (4 bits, avg)")
    plt.savefig('save/save_heatmap_4bits_avg.png', format='png')

    results = np.array([[0.053972649760646904, 0.07710866521120917, 0.1037003076563209, 0.0930713050693524, 0.07777067795636242], \
                        [0.06830002691833212, 0.08308740192594548, 0.0799345894428099, 0.05959464457980756, 0.049686808959701616], \
                        [0.060921288328444, 0.08251452049182824, 0.06136291138892486, 0.0443994947219059, 0.040645205266000524], \
                        [0.08425474229986253, 0.05257203943881278, 0.05788197389648292, 0.057303367510989675, 0.04768184055913403], \
                        [0.05603031322196304, 0.049442282587354815, 0.04777366354922762 , 1, 1]]).transpose((1,0))
    plt.close()
    ax = sns.heatmap(results, xticklabels=conv_layers_list, yticklabels=hidden_size_list, annot=np.round(results*100, 2))
    plt.xlabel("Depth (num conv layer blocks)")
    plt.ylabel("Width (num hidden units)")
    plt.title("Percentage error bits (4 bits, std)")
    plt.savefig('save/save_heatmap_4bits_std.png', format='png')

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # main()
    draw_heatmap()
