#!/bin/bash

seg_filelist='/mnt/115-rbd01/fisheye_dataset/train_seg_gray_mapping.npz'
remapping_table_folder='/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/remapping_table/left'
p2p="${remapping_table_folder}"/remapping_table_panorama2pinholes.npz

script -c \
    "time generate_fisheye_left_truth_from_filelist.py -i ${seg_filelist} --segmentation-gray --panorama-to-3pinholes ${p2p}" \
    generate_fisheye_left_segmentation_gray_from_filelist.log

# # # Test dry run
# time generate_fisheye_left_truth_from_filelist.py \
#     -i ${seg_filelist} \
#     --panorama-to-3pinholes ${p2p} \
#     --segmentation-gray \
#     --dry-run

depth_filelist='/mnt/115-rbd01/fisheye_dataset/train_depth_file_mapping.npz'

script -c \
    "time generate_fisheye_left_truth_from_filelist.py -i ${depth_filelist} --depth --panorama-to-3pinholes ${p2p}" \
    generate_fisheye_left_depth_from_filelist.log

# # # Test dry run
# time generate_fisheye_left_truth_from_filelist.py \
#     -i ${depth_filelist} \
#     --panorama-to-3pinholes ${p2p} \
#     --depth \
#     --dry-run

# remapping_table_folder='/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/remapping_table/left'
# p2p="${remapping_table_folder}"/remapping_table_panorama2pinholes.npz
# seg_filelist='/mnt/115-rbd01/fisheye_dataset/folder_seg_gray_src_to_dst.npz'
# script -c \
#     "time generate_seg_gray_from_npz.py -i ${seg_filelist} --segmentation-gray --panorama-to-3pinholes ${p2p}" \
#     generate_fisheye_left_segmentation_gray_from_filelist_end2end.log
