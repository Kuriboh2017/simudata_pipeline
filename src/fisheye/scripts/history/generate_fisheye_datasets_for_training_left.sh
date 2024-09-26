#!/bin/bash

input_filelist='/mnt/115-rbd01/fisheye_dataset/train_rgb_left_file_mapping.npz'
output_dir='/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/data'
remapping_table_folder='/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/remapping_table/left'
p2f="${remapping_table_folder}"/remapping_table_panorama2fisheye.npz
f2p="${remapping_table_folder}"/remapping_table_fisheye2pinholes.npz

script -c \
    "time generate_fisheye_left_from_filelist.py -i ${input_filelist} -o ${output_dir} --panorama-to-fisheye ${p2f} --fisheye-to-3pinholes ${f2p}" \
    generate_fisheye_left_from_filelist.log

# Test dry run
# time generate_fisheye_left_from_filelist.py \
# -i ${input_filelist} -o ${output_dir} \
# --panorama-to-fisheye ${p2f} --fisheye-to-3pinholes ${f2p} \
# --dry-run
