import os
import cv2
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    root = "/home/vk352/FaceDetection/datasets/test_set/"
    flist = os.listdir(root)
    imgs = [cv2.imread(os.path.join(root, fname)).astype(np.float32) for fname in flist]
    for sig in [20, 25, 30, 40, 50]:
        out_dir = os.path.join("gaussian_noise", f"sigma_{sig}")
        os.makedirs(out_dir, exist_ok=True)
        for fname, img in tqdm(zip(flist, imgs)):
            new_img = (img + np.random.normal(0, sig, img.shape)).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, fname), new_img)
