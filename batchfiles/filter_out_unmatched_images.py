#!/usr/bin/env python3
# In the original dataset,
# group0 is for pinholes.
# group1 is for panorama

from datetime import datetime
from functools import partial
from lz4.frame import compress as lzcompress
from lz4.frame import decompress as lzdecompress
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import csv
import cv2
import itertools
import logging
import lz4.frame as lz
import math
import numpy as np
import os
import pickle as pkl
import re
import warnings

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def _read_lz4(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr


def _get_compressed_data(left_img_path, target):
    paired_path = left_img_path.replace(
        'Image', target).replace('.webp', '.npz')
    if not Path(paired_path).exists():
        paired_path = left_img_path.replace(
            'Image', target).replace('.webp', '.lz4')
        return _read_lz4(paired_path) if Path(paired_path).exists() else None
    else:
        npz_data = np.load(paired_path)
        return npz_data[npz_data.files[0]]


def load_paired_data(left_img_path):
    right_img_path = left_img_path.replace(
        'cam0_0', 'cam0_1').replace('cam1_0', 'cam1_1')
    if any(not Path(p).exists() for p in [left_img_path, right_img_path]):
        return None, None, None
    left_depth = _get_compressed_data(left_img_path, 'Depth')
    left_disp = _get_compressed_data(left_img_path, 'Disparity')
    if left_depth is None or left_disp is None:
        return None, None, None, None
    l_img = cv2.imread(left_img_path)
    r_img = cv2.imread(right_img_path)
    return l_img, r_img, left_depth, left_disp


def is_image_too_close(depth, src_limg_path, depth_threshold_meter=0.2, percent_threshold=0.10):
    '''
    Check if the image is too close to the camera.
    '''
    count = np.sum(depth < depth_threshold_meter)
    total_pixels = depth.shape[0] * depth.shape[1]
    _logger.info(
        f'percent of pixels too close: {count / total_pixels} for {src_limg_path}')
    return count / total_pixels > percent_threshold


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    product /= stds
    return product


def warp(left_img, left_disp):
    H, W, C = left_img.shape
    pad_size = 300
    left_img_pad = cv2.copyMakeBorder(
        left_img, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)).astype(np.float32)
    # r_width, r_height = left_img_pad.shape
    hs = np.linspace(0, H - 1, H).astype(np.int16)
    ws = np.linspace(0, W - 1, W).astype(np.int16)
    wv, hv = np.meshgrid(ws, hs)
    wv_r = wv - left_disp
    wv_r_ceil = np.ceil(wv_r)
    wv_r_floor = np.floor(wv_r)
    dist_to_floor = wv_r - wv_r_floor
    dist_to_floor = np.repeat(np.expand_dims(dist_to_floor, axis=2), C, axis=2)
    left_pad_ceil = left_img_pad[hv+pad_size,
                                 wv_r_ceil.astype(np.int16)+pad_size, :]
    left_pad_floor = left_img_pad[hv+pad_size,
                                  wv_r_floor.astype(np.int16)+pad_size, :]
    light_img = dist_to_floor * left_pad_ceil + \
        (1 - dist_to_floor) * left_pad_floor
    return light_img.astype(np.uint8)


def compare(im1, im2, d=9, sample=9):
    similar, total = 0, 0
    sh_row, sh_col = im1.shape[0], im1.shape[1]
    correlation = np.zeros((int(sh_row/sample), int(sh_col/sample)))
    total = correlation.shape[0] * correlation.shape[1]
    for x, i in enumerate(range(d, sh_row - (d + 1), sample)):
        for y, j in enumerate(range(d, sh_col - (d + 1), sample)):
            correlation[x, y] = correlation_coefficient(im1[i - d: i + d + 1,
                                                            j - d: j + d + 1],
                                                        im2[i - d: i + d + 1,
                                                            j - d: j + d + 1])
            if correlation[x, y] > 0.8:   # Threshold to determine a pixel is similar or not
                similar += 1
    return float(similar/total)


def is_warped_left_image_similar_to_right_image(limg, rimg, disp, src_limg_path, threshold=0.75):
    '''
    Check if the warped left image is similar to the right image.
    '''
    warped_limg = warp(limg, -disp)
    similarity = compare(warped_limg, rimg)
    _logger.info(f'Similarity: {similarity} for {src_limg_path}')
    return similarity > threshold


def process_one(src_limg_path, args):
    try:
        l_img, r_img, left_depth, left_disp = load_paired_data(src_limg_path)
        if l_img is None or r_img is None or left_depth is None or left_disp is None:
            return src_limg_path
        if is_image_too_close(left_depth, src_limg_path):
            _logger.info(f'Image too close: {src_limg_path}')
            return src_limg_path
        if not is_warped_left_image_similar_to_right_image(l_img, r_img, left_disp, src_limg_path):
            _logger.info(f'Image not similar: {src_limg_path}')
            return src_limg_path

    except Exception as e:
        _logger.error(f'Error processing {src_limg_path}: {e}')
        return src_limg_path

    _logger.info(f'Processed {src_limg_path}')
    return None


def get_image_file_list(data_root):
    return [
        str(file_path)
        for file_path in Path(data_root).rglob('*')
        if file_path.is_file()
        and '.webp' in file_path.suffix
        # Pinholes: or 'cam0_0' in str(file_path)
        and ('cam1_0' in str(file_path))
        and 'Image' in str(file_path)
    ]


def delete_related_files(input_dir, file_path):
    timestamp_str = Path(file_path).stem
    target_paths = list(Path(input_dir).rglob(f'{timestamp_str}*'))
    for p in target_paths:
        _logger.info(f'Deleting {p}')
        p.unlink()


def _main(args):
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f'{input_dir} does not exist'
    images = get_image_file_list(input_dir)
    _logger.info(f'Found {len(images)} images in {input_dir}')

    with Pool(cpu_count()) as pool:
        results = pool.map(partial(process_one, args=args), images)

    _logger.info(f'Processed {len(results)} images out of {len(images)}')
    found_images = [r for r in results if r is not None]
    valid_count = len(images) - len(found_images)
    _logger.info(f'Valid images: {valid_count} / {len(results)}')
    if args.delete:
        for file in found_images:
            delete_related_files(input_dir, file)
    else:
        output_file = input_dir / 'unmatched_images.txt'
        with open(str(output_file), 'w') as f:
            f.write('\n'.join(found_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Before running fisheye conversion, '
                     'filter the unmatched panorama images.'
                     )
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='Delete the unmatched images')
    args = parser.parse_args()
    _main(args)
