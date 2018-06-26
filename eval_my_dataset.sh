#!/bin/bash

input_list="../dataset/msp/my_dataset_eval_list.txt"
checkpoint_dir=./checkpoint_my_dataset/
checkpoint=''
save_path=./my_dataset_result.csv

channels=5
eps=1e-10

frame_length=1024
frame_hop=256
stft_frames=128
sampling_rate=8192
overlap=64

python3 ./eval.py --input_list="$input_list" --checkpoint_dir=$checkpoint_dir --checkpoint=$checkpoint --save_path=$save_path --channels=$channels --eps=$eps --frame_length=$frame_length --frame_hop=$frame_hop --stft_frames=$stft_frames --sampling_rate=$sampling_rate --overlap=$overlap
