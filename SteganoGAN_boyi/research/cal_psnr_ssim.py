import argparse
import os
from util import util
import torch
from math import log10

def calc_psnr(img1, img2):
    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)
    
  
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_psnr_and_ssim(img0, img1):
    ### args:
        # img0: pytorch tensor, range [-1, 1]
        # img1: pytorch tensor, range [-1, 1]

    ### prepare data
    img0 = (img0+1.) * 127.5
    img1 = (img1+1.) * 127.5
    if (img0.size() != img1.size()):
        h_min = min(img0.size(2), img1.size(2))
        w_min = min(img0.size(3), img1.size(3))
        img0 = img0[:, :, :h_min, :w_min]
        img1 = img1[:, :, :h_min, :w_min]

    img0 = np.transpose(img0.squeeze().round().cpu().numpy(), (1,2,0))
    img1 = np.transpose(img1.squeeze().round().cpu().numpy(), (1,2,0))

    psnr = calc_psnr(img0, img1)
    ssim = calc_ssim(img0, img1)

    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir0', type=str, default='./imgs/ex_dir0')
    parser.add_argument('--dir1', type=str, default='./imgs/ex_dir1')
    opt = parser.parse_args()

    device = torch.device('cuda')

    files = os.listdir(opt.dir0)
    avg_psnr = 0
    avg_ssim = 0
    for file in files:
        file0 = file
        file1 = file.replace('real', 'fake')
        if(os.path.exists(os.path.join(opt.dir1,file1))):
            img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0, file0))) # RGB image from [-1,1]
            img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1, file1)))
            img0, img1 = img0.to(device), img1.to(device)
            psnr, ssim = calc_psnr_and_ssim(img0.detach(), img1.detach())
            avg_psnr += psnr
            avg_ssim += ssim
            # print(psnr)

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(files)))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim / len(files)))


if __name__ == '__main__':
    main()