#!/bin/bash

input_filelist='/mnt/115-rbd01/fisheye_dataset/train_rgb_right_file_mapping.npz'

script -c \
    "time generate_fisheye_right_from_filelist.py -i ${input_filelist}" \
    generate_fisheye_right_from_filelist.log

# Test dry run
# time generate_fisheye_right_from_filelist.py \
#     -i ${input_filelist} \
#     --dry-run
