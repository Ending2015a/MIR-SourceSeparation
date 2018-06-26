#!/bin/bash

input_file="../dataset/msp/musdb18-2/train/Flags - 54.stem/Flags - 54_0.wav"
checkpoint_dir=./checkpoint_musdb/
checkpoint=./checkpoint_musdb/model.ckpt-44880
save_path=./

channels=4
eps=1e-10

frame_length=1024
frame_hop=256
stft_frames=128
sampling_rate=8192
overlap=64

python3 ./inference.py --input_file="$input_file" --checkpoint_dir=$checkpoint_dir --checkpoint=$checkpoint --save_path=$save_path --channels=$channels --eps=$eps --frame_length=$frame_length --frame_hop=$frame_hop --stft_frames=$stft_frames --sampling_rate=$sampling_rate --overlap=$overlap
