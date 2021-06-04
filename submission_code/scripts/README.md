# Run the hamming code experiments
'''
CUDA_VISIBLE_DEVICES=0 python hamming_result.py  --loss2 BCE_bit_vk  /
--model model0 --steps2 2000 --steps1 2000 --hidden 128 --lr1 0.1 --image_range 100  --number 5 /
--bit 3  --save_path result/hamming/restart_5x_bit_3 --not_early --image_path path-to-the-data
'''
