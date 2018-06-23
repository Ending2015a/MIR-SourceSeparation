#!/bin/bash

input_list=../dataset/msp/musdb2_train_list.txt
log_dir=./log_musdb/
checkpoint_dir=./checkpoint_musdb/
channels=4
stft_frames=128
sampling_rate=8192

batch_size=64
cost_function=l1
eps=1e-10
initial_learning_rate=1e-5
minimum_learning_rate=1e-10
learning_rate_decay_factor=0.1
num_epochs=1000
num_epochs_before_decay=100
optimizer=adam

python3 ./train.py --input_list=$input_list --log_dir=$log_dir --checkpoint_dir=$checkpoint_dir --channels=$channels --batch_size=$batch_size --stft_frames=$stft_frames --cost_function=$cost_function --eps=$eps --initial_learning_rate=$initial_learning_rate --minimum_learning_rate=$minimum_learning_rate --learning_rate_decay_factor=$learning_rate_decay_factor --num_epochs=$num_epochs --num_epochs_before_decay=$num_epochs_before_decay --optimizer=$optimizer --sampling_rate=$sampling_rate
