# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from utils.inverter_encryption import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('image_list', type=str,
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--seed', type=int, default=0,
                      help='random seed')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--mask', type=None, default=None,
                      help='the mask for mse reconstruction')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  #logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')
  logger = setup_logger('')

  torch.manual_seed(args.seed)
  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      logger=logger)
  image_size = inverter.G.resolution
    
  if args.mask is not None:
    loaded_mask = torch.from_numpy(np.load(args.mask)).to("cuda")
    factor = loaded_mask.size()[0] / float(image_size)
    mask = torch.zeros([1, 3, image_size, image_size]).to("cuda")
    for i in range(image_size):
        for j in range(image_size):
            mask[:, :, i, j] = (loaded_mask[int(i * factor): int((i + 1) * factor), int(j * factor) : int((j + 1) * factor)].sum() >= 1).float()
    mask = 1 - mask
#     mask = torch.zeros([1, 3, image_size, image_size]).to("cuda")
#     h1, h2, w1, w2 = args.mask.split("_")
#     h1, h2, w1, w2 = int(float(h1) / 1000 * image_size), int(float(h2) / 1000 * image_size), int(float(w1) / 1000 * image_size), int(float(w2) / 1000 * image_size)
#     print(h1, h2, w1, w2)
#     mask[:, :, :w1] = 1
#     mask[:, :, w2:] = 1
#     mask[:, :, :, :h1] = 1
#     mask[:, :, :, h2:] = 1
  else:
    mask = None

  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  image_list = [args.image_list]
#   with open(args.image_list, 'r') as f:
#     for line in f:
#       image_list.append(line.strip())

  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Invert images.
  logger.info(f'Start inversion.')
  latent_codes = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    image_path = image_list[img_idx]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = resize_image(load_image(image_path), (image_size, image_size))
    code, viz_results = inverter.easy_invert(image, mask=mask, num_viz=args.num_results)
    latent_codes.append(code)
    if mask is not None:
     save_image(f'{output_dir}/{image_name}_ori_{args.seed}.jpg', image)
     save_image(f'{output_dir}/{image_name}_ori_mask_{args.seed}.jpg', viz_results[-1])
     save_image(f'{output_dir}/{image_name}_rec_enc_{args.seed}.jpg', viz_results[1])
     save_image(f'{output_dir}/{image_name}_rec_inv_{args.seed}.jpg', viz_results[-2])
    else:
     save_image(f'{output_dir}/{image_name}_ori_{args.seed}.jpg', image)
     save_image(f'{output_dir}/{image_name}_enc_{args.seed}.jpg', viz_results[1])
     save_image(f'{output_dir}/{image_name}_inv_{args.seed}.jpg', viz_results[-1])
#     visualizer.set_cell(img_idx, 0, text=image_name)
#     visualizer.set_cell(img_idx, 1, image=image)
#     for viz_idx, viz_img in enumerate(viz_results[1:]):
#       visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)

  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()
