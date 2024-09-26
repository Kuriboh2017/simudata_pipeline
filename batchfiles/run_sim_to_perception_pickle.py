#!/usr/bin/env python3

from add_isp import get_random_mode_config
from add_sharpen import get_sharpen_strength
from add_vignette import get_vignette_strength
from datetime import datetime
from functools import partial
from lz4.frame import compress as lzcompress
from lz4.frame import decompress as lzdecompress
from multiprocessing import Pool, cpu_count
from numba import njit
from pathlib import Path
from run_fisheye_isp_with_erp import get_seed_from_filename
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
from PIL import Image

import erp_gt_check_cv

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)


# 将numba的日志等级提高到warning，从而屏蔽njit的超长log
logging.getLogger('numba').setLevel(logging.WARNING)

def _read_lz4(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr


def load_webp_img(left_img_path):
    right_img_path = left_img_path.replace(
        'cam0_0', 'cam0_1').replace('cam1_0', 'cam1_1')
    l_img = cv2.imread(left_img_path)
    r_img = cv2.imread(right_img_path)
    return l_img, r_img


def _rename_isp_config(isp_config):
    # filter id 0: Exposure: increase or decrease the brightness of the image
    # filter id 1: Gamma: increase or decrease the gamma of the image
    # filter id 2: Saturation: increase or decrease the saturation of the image
    # filter id 4: Contrast: increase or decrease the saturation of the image
    # filter id 6: Tone: increase or decrease the tone (shadows, midtones, highlights) of the image
    filter_id_map = {
        0: 'exposure',
        1: 'gamma',
        2: 'saturation',
        4: 'contrast',
        6: 'tone'
    }
    renamed_isp_config = []
    for isp_filter_dict in isp_config:
        filter_id = isp_filter_dict['filter_id']
        parameters = isp_filter_dict['parameters']
        renamed_isp_config.append({
            filter_id_map[filter_id]: parameters})
    return renamed_isp_config


def _get_isp_info(left_img_path, is_left):
    p = str(left_img_path)
    seed1 = get_seed_from_filename(p)
    seed2 = 0 if is_left else 1
    vignette_strength = get_vignette_strength(seed1, seed2)
    sharpen_strength = get_sharpen_strength(seed1, seed2)
    random_isp_config = _rename_isp_config(
        get_random_mode_config(seed1, seed2))
    return {
        'vignette_strength': vignette_strength,
        'sharpen_strength': sharpen_strength,
        'random_isp_config': random_isp_config,
    }


def save_pkl_imgs(out_path, limg, rimg, left_img_path, delete_odd_row=False, calib=-1000):
    lzcompress_rate = 9
    if delete_odd_row:
        limg = np.ascontiguousarray(limg[::2, :, :])
        rimg = np.ascontiguousarray(rimg[::2, :, :])
    
    limg = Image.fromarray(limg)
    rimg = Image.fromarray(rimg)
        
    limg.save(left_img_path, lossless = False)
    right_img_path = left_img_path.replace(
        'cam0_0', 'cam0_1').replace('cam1_0', 'cam1_1')
    rimg.save(right_img_path, lossless = False)
    '''
    # 为了节省存储，不再打包成pkl文件，而是直接存储为webp
    isp_info_left = _get_isp_info(left_img_path, True)
    isp_info_right = _get_isp_info(left_img_path, False)
    with open(out_path, "wb") as f_img_out:
        img_data = {
            'left_image': {'data': lzcompress(limg, lzcompress_rate), 'shape': limg.shape, 'dtype': limg.dtype},
            'right_image': {'data': lzcompress(rimg, lzcompress_rate), 'shape': rimg.shape, 'dtype': rimg.dtype},
            'left_path': left_img_path,
            'calib': calib,
            'default_shape': limg.shape,
            'info': {'isp': {'left': isp_info_left, 'right': isp_info_right}}
        }
        pkl.dump(img_data, f_img_out)
    '''

def fdelete_odd_row(img):
    # 为了防止关键信息（比如细小物体信息）被删掉，采用比较两行取更近的像素
    return np.maximum(img[::2,:], img[1::2,:])
        

def save_pkl_disp(out_path, disp, tiny_mask, foreground_mask, confusing_mask, baseline, delete_odd_row=False):
    lzcompress_rate = 9     # 默认压缩等级
    if delete_odd_row:
        # 为了防止关键信息（比如细小物体信息）被删掉，采用比较两行取更近的像素
        disp = np.ascontiguousarray(fdelete_odd_row(disp))
        tiny_mask = np.ascontiguousarray(fdelete_odd_row(tiny_mask))
        foreground_mask = np.ascontiguousarray(fdelete_odd_row(foreground_mask))
        confusing_mask = np.ascontiguousarray(fdelete_odd_row(confusing_mask))
    error = np.zeros_like(disp)
    with open(out_path, "wb") as f_disp_out:
        disp_data = {
            'left_disparity': {'data': lzcompress(disp, lzcompress_rate), 'shape': disp.shape, 'dtype': disp.dtype},
            'tiny_mask': {'data': lzcompress(tiny_mask, lzcompress_rate), 'shape': tiny_mask.shape, 'dtype': tiny_mask.dtype},
            'foreground_mask': {'data': lzcompress(foreground_mask, lzcompress_rate), 'shape': foreground_mask.shape, 'dtype': foreground_mask.dtype},
            'confusing_mask': {'data': lzcompress(confusing_mask, lzcompress_rate), 'shape': confusing_mask.shape, 'dtype': confusing_mask.dtype},
            'default_shape': disp.shape,
            'baseline': baseline,
        }
        if not delete_odd_row:
            disp_data['left_disparity_err'] = {
                'data': lzcompress(error, lzcompress_rate),
                'shape': error.shape,
                'dtype': error.dtype,
            }
        pkl.dump(disp_data, f_disp_out)


def save_pkl_segs(out_path, seg, delete_odd_row=False):
    lzcompress_rate = 9
    if delete_odd_row:
        seg = np.ascontiguousarray(fdelete_odd_row(seg))
    with open(out_path, "wb") as f_seg_out:
        seg_data = {
            'segmentation': {'data': lzcompress(seg, lzcompress_rate), 'shape': seg.shape, 'dtype': seg.dtype},
            'default_shape': seg.dtype
        }
        pkl.dump(seg_data, f_seg_out)


def output_paths(output_dir, in_limg_path):

    kwd_pattern = 'group'
    path_split = in_limg_path.split('/')

    group_idx = -1
    for item_idx, item in enumerate(path_split):
        if re.match(kwd_pattern, item):
            group_idx = item_idx
            break

    out_imgs_folder = os.path.join(
        *([str(output_dir / 'Images')] + path_split[group_idx-1: group_idx+3]))
    #out_imgs_folder = str(Path(out_imgs_folder).parent / 'Image_3to1')
    out_imgs_folder = str(Path(out_imgs_folder).parent / 'Image_erp')
    out_disp_folder = out_imgs_folder.replace('Images', 'Disparity')
    out_segs_folder = out_imgs_folder.replace('Images', 'Segment')

    os.makedirs(out_imgs_folder, exist_ok=True)
    os.makedirs(out_disp_folder, exist_ok=True)
    os.makedirs(out_segs_folder, exist_ok=True)

    # out_imgs_path = os.path.join(
    #     out_imgs_folder, path_split[-1]).replace('.webp', '.pkl')
    # img不再压缩为pkl文件，而是直接存储为webp
    out_imgs_path = os.path.join(
        out_imgs_folder, path_split[-1])
    out_disp_path = os.path.join(
        out_disp_folder, path_split[-1]).replace('.webp', '.pkl')
    out_segs_path = os.path.join(
        out_segs_folder, path_split[-1]).replace('.webp', '.pkl')

    return out_imgs_path, out_disp_path, out_segs_path

def disp2depth(disp_data, baseline):
    # 使用float32避免溢出(overflow)
    disp_data = disp_data.astype(np.float32)
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    p = [3.6892000000000002e+02, 3.6892000000000002e+02,
         6.3950000000000000e+02, 2.2650000000000000e+02,]
    zDatas = np.zeros([ori_input_h, ori_input_w], dtype=np.float32)
    partZDatas = np.zeros([avg_input_h, ori_input_w], dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to be treated as errors
        warnings.simplefilter("error")
        try:
            # RuntimeWarning: overflow encountered in divide
            #  partZDatas = baseline * p[0] / partDisp
            for k in range(3):
                k_start = k * avg_input_h
                k_end = k_start + avg_input_h
                partDisp = disp_data[k_start:k_end, :]
                partZDatas = baseline * p[0] / partDisp
                for j, i in itertools.product(range(avg_input_h), range(ori_input_w)):
                    x = math.pow((i - p[2]), 2) / math.pow(p[0], 2)
                    y = math.pow((j - p[3]), 2) / math.pow(p[1], 2)
                    partDepth = math.sqrt(
                        math.pow(partZDatas[j, i], 2) * (x + y + 1))
                    zDatas[j + k_start, i] = partDepth
        except RuntimeWarning as e:
            _logger.error(f'Error processing {e}')
            return None
    return zDatas

def disp2depth_erp(disp, baseline, max_depth=10000, depthGT=None):
    # 1408x1920的内参
    cx=639.5
    cy=703.5
    fx=611.1549814
    fy=436.0673381
    baseline = abs(baseline)
    H, W = disp.shape
    grid = np.mgrid[0:H, 0:W]
    v, u = grid[0], grid[1] # u = [0, W-1], v = [0, H-1]
    theta_l_map = (u - cx) / fx
    phi_l_map = (v - cy) / fy

    disp_rad = disp / fx
    disp_rad = np.clip(disp_rad, a_min=0, a_max=theta_l_map+np.pi/2) # physicial restriction on disparity
    mask_disp_is_0 = (disp_rad == 0) # | (np.isnan(disp)) # there is no nan in disp from the network
    disp_not_0_rad = np.ma.array(disp_rad, mask=mask_disp_is_0)
    theta_r_map = theta_l_map - disp_not_0_rad # range from -np.pi/2 to theta_l_map

    if depthGT is None:
        # depth = baseline * np.sin(theta_r_map) / np.sin(disp_not_0_rad)
        depth = baseline * np.sin(np.pi / 2 + theta_r_map) / np.sin(disp_not_0_rad)

        # handle special pixels
        depth = depth.filled(max_depth)
    else:
        depth = depthGT
    depth[depth > max_depth] = max_depth
    return depth

# 根据深度提取细小物体
def get_thinmask_by_depth(depth_img):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to be treated as errors
        warnings.simplefilter("error")
        try:
            depth = depth_img.copy()
            far_value = 30  # 只关心30m以内的细小物体
            far_mask = depth > far_value
            threshold = 0.7  # 分辨细小物体边界的阈值，反复测试确定
            rela_threshold = 0.3    # 相对误差阈值，对于更近处物体，分辨细小物体的阈值
            # 用于膨胀腐蚀的核，细小物体的定义为8个像素，故核的大小为9
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            # 2024.3.7 新需求，tiny核扩大到17
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))
            
            closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
            diff = np.abs(depth - closing)
            tiny_mask = diff > threshold
            tiny_mask[far_mask] = False
            # 计算一个相对误差范围
            tiny_mask_rela = tiny_mask | (diff/(depth+1e-3)>rela_threshold)
            tiny_mask_rela[far_mask] = False
        except RuntimeWarning as e:
            _logger.error(f'Error processing {e}')
            return None
    return tiny_mask_rela

# 获得前景mask,其实就是扩大的tiny_mask
def get_foreground_mask_by_depth(depth_img):
    depth = depth_img.copy()
    far_value = 30  # 只关心30m以内的细小物体
    far_mask = depth > far_value    
    threshold = 0.7
    kernel_foreground_size = 45
    kernel_foreground = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_foreground_size, kernel_foreground_size))
    closing_foreground = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel_foreground)
    diff_foreground = np.abs(depth - closing_foreground)
    foreground_mask = diff_foreground > threshold
    foreground_mask[far_mask] = False
    return foreground_mask

def get_confusing_mask_by_tiny_depth(tiny_mask,l_img):
    # confusion mask
    #depth = depth_img.copy()
    rgb_thresh=25
    rgb_image = l_img.copy()
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
    # pdb.set_trace()
    confusion_mask = (rgb_magnitude < rgb_thresh)
    tiny_confusion_mask = tiny_mask & confusion_mask
    return tiny_confusion_mask
    
def get_tiny_confusion_mask_by_depth(depth_img,l_img):
    '''根据深度提取细小物体,前景,混淆msk,depth_img必须为深度图
    这是真实数据用的源代码，仿真不用这个函数，仅备用参考'''
    rgb_thresh=25
    far_value = 30  # 只关心30m以内的细小物体
    rela_value = 0.3
    threshold = 0.7  # 分辨细小物体边界的阈值，反复测试确定
    depth_src = depth_img.copy()
    depth = depth_img.copy()
    # 真实数据需要模糊和锐化处理
    # 高斯模糊
    depth_blurred_size = 9
    depth_blurred = cv2.GaussianBlur(depth, (depth_blurred_size, depth_blurred_size), 0)
    # 锐化
    depth_sharpen_size = 3
    filter = np.ones((depth_sharpen_size,depth_sharpen_size))*-1
    filter[depth_sharpen_size//2,depth_sharpen_size//2] = depth_sharpen_size**2
    # pdb.set_trace()
    depth=cv2.filter2D(depth_blurred,-1,filter)
    far_mask = depth > far_value
    # 用于膨胀腐蚀的核，细小物体的定义为8个像素，故核的大小为9
    kernel_size_value = 17
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_value, kernel_size_value))
    closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    diff = np.abs(depth - closing)
    tiny_mask = diff > threshold
    tiny_mask[far_mask] = False
    # 计算一个相对误差范围
    tiny_mask_rela = tiny_mask | (diff/(depth+1e-3)>rela_value)
    tiny_mask_rela[far_mask] = False

    # 前景mask
    # kernel_foreground_size = kernel_size_value*2+1
    kernel_foreground_size = 45
    kernel_foreground = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_foreground_size, kernel_foreground_size))
    closing_foreground = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel_foreground)
    diff_foreground = np.abs(depth - closing_foreground)
    foreground_mask = diff_foreground > threshold
    foreground_mask[far_mask] = False
    # confusion mask
    rgb_image = l_img
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
    # pdb.set_trace()
    confusion_mask = (rgb_magnitude < rgb_thresh)
    tiny_confusion_mask = tiny_mask & confusion_mask

    return tiny_mask_rela,tiny_confusion_mask,foreground_mask

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


def save_output_data(output_directory, src_img_path, limg, rimg, disp, tiny_mask, foreground_mask, confusing_mask, segs, baseline, delete_odd_row=False):
    out_imgs, out_disp, out_segs = output_paths(output_directory, src_img_path)
    save_pkl_imgs(out_imgs, limg, rimg, src_img_path,
                  delete_odd_row=delete_odd_row)
    save_pkl_disp(out_disp, disp, tiny_mask, foreground_mask, confusing_mask, baseline,
                  delete_odd_row=delete_odd_row)
    save_pkl_segs(out_segs, segs, delete_odd_row=delete_odd_row)
    return out_imgs, out_disp, out_segs

def is_image_too_close(depth, depth_threshold_meter=0.3, percent_threshold=0.10):
    '''
    Check if the image is too close to the camera.
    '''
    count = np.sum(depth < depth_threshold_meter)
    total_pixels = depth.shape[0] * depth.shape[1]
    return count / total_pixels > percent_threshold

@njit    # Thanks to Xuanquan, njit 可以加速数学计算，原来6分钟的计算过程现在只要十几秒
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


def get_warped_left_image_similarity_to_right_image(limg, rimg, disp):
    '''
    Get the similarity between warped left image and the right image.
    '''
    warped_limg = warp(limg, -disp)
    return compare(warped_limg, rimg)

def _remap_segmentation_id(seg_graymap):
    seg_graymap = seg_graymap.copy()
    # seg_graymap[seg_graymap == 1] = 0
    # seg_graymap[seg_graymap == 2] = 0
    seg_graymap[seg_graymap == 0] = 1
    seg_graymap[seg_graymap == 11] = 0
    seg_graymap[seg_graymap == 18] = 2
    return seg_graymap

def fix_segment(seg, disp):
    mask = disp > 0.1
    new_seg = seg.copy()
    new_seg[mask] = 1
    return new_seg

def process_one(src_limg_path, args):
    output_dir = Path(args.output_dir)
    src_disp_path = src_limg_path.replace(
        'Image', 'Disparity').replace('.webp', '.lz4')
    src_segs_path = src_limg_path.replace(
        'Image', 'Segmentation/Graymap').replace('.webp', '.lz4')

    files = [src_limg_path, src_disp_path, src_segs_path]
    if not all(os.path.exists(f) for f in files):
        _logger.error(f'Not all files exist: {files}')
        return False

    try:
        limg, rimg = load_webp_img(src_limg_path)
        disp = _read_lz4(src_disp_path)
        segs = _read_lz4(src_segs_path)

        #print(f'size:{limg.shape}||{rimg.shape}||{disp.shape}||{segs.shape}')

        # 处理分割图标签，之前没有转化成0,1,2，在这里转一下
        segs = _remap_segmentation_id(segs)

        # 临时将细小物体分割图的电线设为255
        # seg255 = segs.copy()
        # seg255[segs == 2] = 255
        # segs = seg255

        #dismiss nan
        if disp is None or np.isnan(disp).any():
            #_logger.error(f'Disparity has nan:{src_limg_path}')
            # 先基于天空的分割把nan的视差设为0, 注意这里的分割值不是0，1，2，是0，11，18
            sky_mask = segs == 0
            nan_mask = np.isnan(disp)
            disp_cp=disp.copy()
            disp_cp[sky_mask & nan_mask] = 0
            disp = disp_cp
            # 二次检测nan,丢弃仍有nan的数据
            if np.isnan(disp).any():
                return False

        # Generate required masks
        #baseline = 0.09 if 'group0/cam0_0' in src_limg_path else 0.105
        if 'group0/cam0_0' in src_limg_path:
            if 'MX128' in src_limg_path:
                baseline = 0.128
            else:
                baseline = 0.09
        else:
            baseline = 0.105
        
        #depth = disp2depth(disp, baseline)
        depth = disp2depth_erp(disp, baseline)  # 现在用erp转深度
        if depth is None:
            return False
        if is_image_too_close(depth):
            _logger.info(f'Image too close: {src_limg_path}')
            return False
        if args.filter_unmatched:
            similarity = get_warped_left_image_similarity_to_right_image(limg, rimg, disp)
            _logger.info(f'Similarity: {similarity} for {src_limg_path}')
            if similarity < 0.6:
                _logger.info(f'Stereo images are not similar: {src_limg_path}')
                return False

        tiny_mask = get_thinmask_by_depth(depth)
        foreground_mask = get_foreground_mask_by_depth(depth)
        
        if tiny_mask is None:
            return False
        #confusing_mask = get_confusing_pixels(limg, depth)
        confusing_mask = get_confusing_mask_by_tiny_depth(tiny_mask,limg)

        # Save the original data
        out_imgs, out_disp, out_segs = save_output_data(output_dir, src_limg_path, limg, rimg,
                                                        disp, tiny_mask, foreground_mask, confusing_mask, segs, baseline)

        # Save the downsampled data if the downsampled output directory is provided
        if args.downsampled_output_dir:
            downsampled_output_dir = Path(args.downsampled_output_dir)
            downsampled_out_imgs, downsampled_out_disp, downsampled_out_segs = \
                save_output_data(downsampled_output_dir, src_limg_path, limg, rimg,
                                 disp, tiny_mask, foreground_mask, confusing_mask, segs, baseline, delete_odd_row=True)

    except Exception as e:
        _logger.error(f'Error processing {src_limg_path}: {e}')
        return False

    _logger.info(f'Processed {src_limg_path}')
    output = (
        str(Path(out_imgs).relative_to(output_dir)),
        str(Path(out_disp).relative_to(output_dir)),
        str(Path(out_segs).relative_to(output_dir))
    )
    downsampled_output = None
    if args.downsampled_output_dir:
        downsampled_output = (
            str(Path(downsampled_out_imgs).relative_to(downsampled_output_dir)),
            str(Path(downsampled_out_disp).relative_to(downsampled_output_dir)),
            str(Path(downsampled_out_segs).relative_to(downsampled_output_dir))
        )
    return output, downsampled_output

def get_image_file_list(data_root):
    return [
        str(file_path)
        for file_path in Path(data_root).rglob('*')
        if file_path.is_file()
        and '.webp' in file_path.suffix
        and ('cam1_0' in str(file_path) or 'cam0_0' in str(file_path))
        and 'Image' in str(file_path)
        and '_3to1/' not in str(file_path)     # 准备不用1拆3，用erp了
        #and '_erp/' not in str(file_path)
    ]


def generate_filelist(output_dir, results, downsampled_results=False):
    num_valid = 0
    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    filelist_dir = output_dir / 'file_list'
    filelist_dir.mkdir(parents=True, exist_ok=True)
    filelist_path = filelist_dir / f'pkl_filelist_fisheye_{time_str}.csv'
    with open(filelist_path, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow([str(output_dir.absolute())] * 3)
        for item in results:
            if item:
                num_valid += 1
                paths = item[1] if downsampled_results else item[0]
                writer.writerow(paths)
    print('downsampled_results = ', downsampled_results)
    print('Valid rate (%d/%d): %.2f' %
          (num_valid, len(results), num_valid/len(results)))


def _main(args):
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f'{input_dir} does not exist'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = get_image_file_list(input_dir)
    _logger.info(f'Found {len(images)} images in {input_dir}')

    with Pool(cpu_count()) as pool:
        results = pool.map(partial(process_one, args=args), images)

    _logger.info(f'Processed {len(results)} images out of {len(images)}')

    # generate filelist and check the number of successfully processed files
    generate_filelist(output_dir, results)
    if args.downsampled_output_dir:
        downsampled_output_dir = Path(args.downsampled_output_dir)
        generate_filelist(downsampled_output_dir, results,
                          downsampled_results=True)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(
        description=('After running all conversion, '
                     'post process images to generate the pickle files.'
                     )
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir',
                        help='Root Directory of the output images')
    parser.add_argument('-d', '--downsampled-output-dir',
                        help='Root Directory of the downsampled output images')
    parser.add_argument('-f', '--filter-unmatched', action='store_true',
                        help='Filter out the unmatched images')
    args = parser.parse_args()
    _main(args)

