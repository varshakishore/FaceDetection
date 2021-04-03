#!/usr/bin/env bash

# Train baseline
# MNIST(0,1,2,3,4) <-> SVHN(2,3,4,5,6)
# Results are stored in outputs/mnist_svhn_baseline_cls1_censor0_nsgan
# From top to bottom: input, translated input, cycle reconstruction
# You should observe that all "5" and "6" of SVHN digits are not correctly translated.
#python train.py --config configs/mnist_svhn_baseline_cls1_censor0_nsgan.yaml --trainer CycleGAN
#python train_new.py --config configs/mnist_usps_baseline_cls1_censor1_nsganCopy.yaml
#python train_new.py --config configs/mnist_usps_baseline_cls1_censor1_nsganVectorCopy.yaml --trainer CycleVectorGAN

python train_boyi.py --config debug_DEVM_celebA.yaml --pretrain 1
