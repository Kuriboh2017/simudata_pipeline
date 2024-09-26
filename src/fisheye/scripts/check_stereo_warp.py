#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def warp(right_img, left_disp):
    H, W, C = right_img.shape
    pad_size = 300
    right_img_pad = cv2.copyMakeBorder(
        right_img, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)).astype(np.float32)
    # r_width, r_height = right_img_pad.shape
    hs = np.linspace(0, H - 1, H).astype(np.int16)
    ws = np.linspace(0, W - 1, W).astype(np.int16)
    wv, hv = np.meshgrid(ws, hs)
    wv_r = wv - left_disp
    wv_r_ceil = np.ceil(wv_r)
    wv_r_floor = np.floor(wv_r)
    dist_to_floor = wv_r - wv_r_floor
    dist_to_floor = np.repeat(np.expand_dims(dist_to_floor, axis=2), C, axis=2)
    right_pad_ceil = right_img_pad[hv+pad_size,
                                   wv_r_ceil.astype(np.int16)+pad_size, :]
    right_pad_floor = right_img_pad[hv+pad_size,
                                    wv_r_floor.astype(np.int16)+pad_size, :]
    light_img = dist_to_floor * right_pad_ceil + \
        (1 - dist_to_floor) * right_pad_floor
    return light_img.astype(np.uint8)


def warp_depth(right_img, left_disp):
    assert right_img.shape == left_disp.shape, \
        "right_img.shape != left_disp.shape"
    H, W = right_img.shape
    pad_size = 300
    right_img_pad = cv2.copyMakeBorder(
        right_img, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT, value=(0, 0)).astype(np.float32)
    hs = np.linspace(0, H - 1, H).astype(np.int16)
    ws = np.linspace(0, W - 1, W).astype(np.int16)
    wv, hv = np.meshgrid(ws, hs)
    wv_r = wv - left_disp
    wv_r_ceil = np.ceil(wv_r)
    wv_r_floor = np.floor(wv_r)
    dist_to_floor = wv_r - wv_r_floor
    right_pad_ceil = right_img_pad[hv+pad_size,
                                   wv_r_ceil.astype(np.int16)+pad_size]
    right_pad_floor = right_img_pad[hv+pad_size,
                                    wv_r_floor.astype(np.int16)+pad_size]
    light_img = dist_to_floor * right_pad_ceil + \
        (1 - dist_to_floor) * right_pad_floor
    return light_img


def _read_lz4(path):
    import lz4.frame as lz
    import pickle as pkl
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr


def check_warp_image(left_scene, right_scene, disparity, output_path):
    scene0_path = left_scene
    scene1_path = right_scene
    if disparity.endswith('.lz4'):
        disparity = _read_lz4(disparity)
    elif disparity.endswith('.npz'):
        disparity = np.load(disparity)['arr_0']
    else:
        raise NotImplementedError
    scene0 = cv2.imread(scene0_path)
    scene1 = cv2.imread(scene1_path)
    print(f'scene0.shape = {scene0.shape}')
    print(f'scene1.shape = {scene1.shape}')
    print(f'disparity.shape = {disparity.shape}')
    warp_img1 = warp(scene0, -disparity)
    if output_path is None:
        warp_img1_path = 'warp_left_to_right.webp'
        warp_img1_path = Path(warp_img1_path).with_suffix(Path(scene0_path).suffix)
    else:
        warp_img1_path = output_path
    cv2.imwrite(str(warp_img1_path), warp_img1)
    error = warp_img1 * 1.0 - scene1 * 1.0
    plt.clf()
    plt.imshow(error)
    plt.colorbar()
    plt.show()
    print(
        f'max, min, average = {np.max(error)}, {np.min(error)}, {np.average(error)}')


def _main(args):
    left_scene = args.l
    right_scene = args.r
    disparity = args.d
    output_path = args.output_path
    check_warp_image(left_scene, right_scene, disparity, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='check stereo image by warp')
    parser.add_argument('-l', required=True,
                        help='left input image')
    parser.add_argument('-r', required=True,
                        help='right input image')
    parser.add_argument('-d', required=True,
                        help='disparity')
    parser.add_argument('-o', '--output-path',
                        help='output path')
    args = parser.parse_args()
    _main(args)
