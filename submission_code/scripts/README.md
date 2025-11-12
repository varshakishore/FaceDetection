# Run the hamming code experiments
```
CUDA_VISIBLE_DEVICES=0 python hamming_result.py  --loss2 BCE_bit_vk  /
--model model0 --steps2 2000 --steps1 2000 --hidden 128 --lr1 0.1 --image_range 100  --number 5 /
--bit 3  --save_path result/hamming/restart_5x_bit_3 --not_early --image_path path-to-the-data
```

# Run the without hamming code experiments
If using the Div2K dataset, the resize function in utils.py can be used to generate the images used for the paper results.
The CelebA and MSCOCO images were used directly from the public datasets.
```
CUDA_VISIBLE_DEVICES=0 python rnns_results.py  --name name_of_image --dataset_path path_to_dataset /
--output_image_path path_to_output_image --num_bits 4 /
```
