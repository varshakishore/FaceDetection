#! /usr/bin/env python
import os
import cv2
import argparse
import numpy as np

from face_detection import select_face
from face_swap import face_swap, mask_from_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', default=None, help='Path for source image')
    parser.add_argument('--dst', default=None, help='Path for target image')
    parser.add_argument('--out', default=None, help='Path for storing output images')
    parser.add_argument('--mask_out', default=None, help='Path for storing src images mask')
    parser.add_argument('--inpainting', default="None", type=str, help='the way to inpainting the inside of ')
    parser.add_argument('--end', default=48, type=int, help='Using how many landmarks for mapping the swapping')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()

    # Select dst face
    dst_img = cv2.imread(args.dst)
    dst_points, dst_shape, dst_face = select_face(dst_img)

    if args.mask_out is not None:
        x, y, w, h = dst_shape
        mask = np.zeros([dst_img.shape[0], dst_img.shape[1]])
        mask[y:y + h, x:x + w] = mask_from_points((h, w), dst_points[:args.end])
        np.save(args.mask_out, mask)
        exit()
        
    # Select src face
    src_img = cv2.imread(args.src)
    src_points, src_shape, src_face = select_face(src_img)

    if src_points is None or dst_points is None:
        print('Detect 0 Face !!!')
        exit(-1)

    output = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args, end=args.end, inpainting=args.inpainting)

    dir_path = os.path.dirname(args.out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args.out, output)

    ##For debug
    if not args.no_debug_window:
        cv2.imshow("From", dst_img)
        cv2.imshow("To", output)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
