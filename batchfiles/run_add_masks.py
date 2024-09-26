#!/usr/bin/env python3

import argparse
import itertools
import math
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging
import cv2
import numpy as np
import os
import re

from lz4.frame import compress as lzcompress
from lz4.frame import decompress as lzdecompress
import pickle as pkl

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def disp2depth(disp_data, baseline):
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    p = [3.6892000000000002e+02, 3.6892000000000002e+02,
         6.3950000000000000e+02, 2.2650000000000000e+02,]
    zDatas = np.zeros([ori_input_h, ori_input_w], dtype=np.float32)
    partZDatas = np.zeros([avg_input_h, ori_input_w], dtype=np.float32)
    for k in range(3):
        k_start = k * avg_input_h
        k_end = k_start + avg_input_h
        partDisp = disp_data[k_start:k_end, :]
        partZDatas = baseline * p[0] / partDisp
        for j, i in itertools.product(range(avg_input_h), range(ori_input_w)):
            x = math.pow((i - p[2]), 2) / math.pow(p[0], 2)
            y = math.pow((j - p[3]), 2) / math.pow(p[1], 2)
            partDepth = math.sqrt(math.pow(partZDatas[j, i], 2) * (x + y + 1))
            zDatas[j + k_start, i] = partDepth
    return zDatas

# Old format: 2x/ folder
# def load_pkl_imgs(img_path):
#     with open(img_path, "rb") as f_img_in:
#         img_data = pkl.load(f_img_in)
#         limg = np.frombuffer(lzdecompress(
#             img_data['left_image']), dtype=img_data['image_dtype'])
#         limg = limg.reshape(img_data['image_shape'])
#     return limg


# def load_pkl_disp(disp_path):
#     with open(disp_path, "rb") as f_disp_in:
#         disp_data = pkl.load(f_disp_in)
#         disp = np.frombuffer(lzdecompress(
#             disp_data['left_disparity']), dtype=disp_data['disparity_dtype'])
#         disp = disp.reshape(disp_data['disparity_shape'])
#     return disp


# New format: 2x_0906/ folder
def load_pkl_imgs(img_path):
    with open(img_path, "rb") as f_img_in:
        img_data = pkl.load(f_img_in)
        limg = np.frombuffer(lzdecompress(
            img_data['left_image']['data']), dtype=img_data['left_image']['dtype'])
        limg = limg.reshape(img_data['left_image']['shape'])
    return limg


def load_pkl_disp(disp_path):
    with open(disp_path, "rb") as f_disp_in:
        disp_data = pkl.load(f_disp_in)
        disp = np.frombuffer(lzdecompress(
            disp_data['left_disparity']['data']), dtype=disp_data['left_disparity']['dtype'])
        disp = disp.reshape(disp_data['left_disparity']['shape'])
    return disp


def assemble_paths(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    root_folders = [folder.strip() for folder in lines[0].split(',')]
    assembled_paths_2d = []
    for line in lines[1:]:
        filenames = [filename.strip() for filename in line.split(',')]
        current_paths = [f"{root_folder}/{filename}" for root_folder,
                         filename in zip(root_folders, filenames)]
        assembled_paths_2d.append(current_paths)
    return assembled_paths_2d


# 根据深度提取细小物体
def get_thinmask_by_depth(depth_img):
    depth = depth_img.copy()
    far_value = 30  # 只关心30m以内的细小物体
    far_mask = depth > far_value
    threshold = 0.7  # 分辨细小物体边界的阈值，反复测试确定
    # 用于膨胀腐蚀的核，细小物体的定义为8个像素，故核的大小为9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    diff = np.abs(depth - closing)
    thin_mask = diff > threshold
    thin_mask[far_mask] = False
    return thin_mask


def get_confusing_pixels(rgb_image, depth_image, rgb_thresh=18, depth_grad_thresh=8, max_depth=30):
    '''
    Find pixels where RGB is similar (below the threshold) and
    depth changes sharply (above the threshold) and depth value is less than max_depth.
    '''
    grad_x_depth = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_depth = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    depth_grad_magnitude = cv2.magnitude(grad_x_depth, grad_y_depth)
    grad_x_r = cv2.Sobel(rgb_image[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_r = cv2.Sobel(rgb_image[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
    grad_x_g = cv2.Sobel(rgb_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_g = cv2.Sobel(rgb_image[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
    grad_x_b = cv2.Sobel(rgb_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_b = cv2.Sobel(rgb_image[:, :, 2], cv2.CV_64F, 0, 1, ksize=3)
    r_magnitude = cv2.magnitude(grad_x_r, grad_y_r)
    g_magnitude = cv2.magnitude(grad_x_g, grad_y_g)
    b_magnitude = cv2.magnitude(grad_x_b, grad_y_b)
    rgb_magnitude = np.sqrt(np.square(r_magnitude) +
                            np.square(g_magnitude) + np.square(b_magnitude))
    depth_binary = (depth_grad_magnitude > depth_grad_thresh).astype(np.uint8)
    depth_value_mask = (depth_image < max_depth).astype(np.uint8)
    depth_binary = cv2.bitwise_and(depth_binary, depth_value_mask)
    rgb_binary = (rgb_magnitude < rgb_thresh).astype(np.uint8)
    return cv2.bitwise_and(rgb_binary, depth_binary)


def _process_one(files):
    assert len(files) == 3, f'Expecting 3 files, got {len(files)}'
    rgb_file, disparity_file, seg_file = files
    mask_file = rgb_file.replace('/Images/', '/Masks/')
    folder = Path(mask_file).parent
    folder.mkdir(parents=True, exist_ok=True)
    rgb = load_pkl_imgs(rgb_file)
    disparity = load_pkl_disp(disparity_file)
    baseline = 0.09 if 'group0/cam0_0' in rgb_file else 0.105
    depth = disp2depth(disparity, baseline)
    thin_mask = get_thinmask_by_depth(depth)
    confusing_mask = get_confusing_pixels(rgb, depth)
    #
    lzcompress_rate = 9
    with open(mask_file, "wb") as f_img_out:
        img_data = {
            'tiny_mask': {'data': lzcompress(thin_mask, lzcompress_rate), 'shape': thin_mask.shape, 'dtype': thin_mask.dtype},
            'confusing_mask': {'data': lzcompress(confusing_mask, lzcompress_rate), 'shape': confusing_mask.shape, 'dtype': confusing_mask.dtype},
        }
        pkl.dump(img_data, f_img_out)
    _logger.info(f'Processed {rgb_file}')


def _main(args):
    file_list_path = args.filelist
    assert os.path.exists(file_list_path), f'File not found: {file_list_path}'
    filepaths_2d = assemble_paths(file_list_path)

    _logger.info(f'Found {len(filepaths_2d)} groups of images.')

    with Pool(cpu_count()) as pool:
        pool.map(partial(_process_one), filepaths_2d)

    _logger.info('Processed all images!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process CSV file with file paths.")
    parser.add_argument('-f', '--filelist',
                        help="Path to the CSV file containing file paths.")
    args = parser.parse_args()
    _main(args)
