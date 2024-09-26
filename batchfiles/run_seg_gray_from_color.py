#!/usr/bin/env python3
import argparse
import cv2
import logging
import lz4.frame as lz
import numpy as np
import pickle as pkl
import os

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image

from run_panorama_disparity_from_depth import get_thinmask_by_depth

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)

# 颜色是BGR顺序，pinhole和fisheye颜色不完全一致是由于没有统一伽马矫正，所以需要统一映射到target_color_map
sim_pinhole_color_map = {'sky': [191, 105, 112], 'wire': [6, 108, 153], 'water': [64,225,190]}
sim_fisheye_color_map = {'sky': [224, 172, 177], 'wire': [42, 174, 203], 'water': [137,241,224]}
target_color_map = {'sky': [70, 130, 180], 'water': [137,241,224],
                    'wire': [126, 96, 247], 'unlabeled': [0, 0, 0]}     
target_gray_map = {'sky': 0, 'wire': 2, 'unlabeled': 1, 'water': 4}


def _save_lz4(image_data, path):
    arr = np.ascontiguousarray(image_data)
    data = {
        'arr': lz.compress(arr, compression_level=3),
        'shape': image_data.shape,
        'dtype': image_data.dtype
    }
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def _read_lz4(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr

def exec_with_scale(func, files):
    with Pool(cpu_count()) as p:
        p.map(func, files)


def _process_segmentation_image(file, color_dir, gray_dir):
    _logger.info(f'Processing {file}')
    img = cv2.imread(str(file))
    color_img = np.zeros_like(img, dtype=np.uint8)
    gray_img = np.zeros_like(img[..., 0], dtype=np.uint8)
    unmatched_mask = np.ones_like(gray_img, dtype=bool)
    # 区分针孔和全景
    if 'left\Segmentation' in str(file):
        color_map = sim_pinhole_color_map
    else:
        color_map = sim_fisheye_color_map
        
    for label, color in color_map.items():
        mask = (img == color).all(axis=-1)
        color_img[mask] = target_color_map[label]
        gray_img[mask] = target_gray_map[label]
        unmatched_mask &= ~mask
    color_img[unmatched_mask] = target_color_map['unlabeled']
    gray_img[unmatched_mask] = target_gray_map['unlabeled']
    color_img = Image.fromarray(color_img)
    color_img.save(str(Path(color_dir) / f'{file.stem}.webp'), lossless=True)
    
    # 为了更好处理细小物体，用膨胀腐蚀放大一点
    # 根据深度图获取细小物体的mask    
    depth_dir = get_depth_path(file)
    if depth_dir is not None and os.path.exists(depth_dir):
        depth_img = _read_lz4(depth_dir)
        tiny_mask = get_thinmask_by_depth(depth_img)
        gray_img =extra_process_gray_seg(gray_img, tiny_mask)
        
    
    _save_lz4(gray_img, str(Path(gray_dir) / f'{file.stem}.lz4'))
    file.unlink()
    _logger.info(f'Processed {file}')

def get_depth_path(seg_path):
    if 'cube_front' in str(seg_path):
        depth_dir = str(seg_path).replace('CubeSegmentation', 'CubeDepth')        
    elif 'left' in str(seg_path):
        depth_dir = str(seg_path).replace('Segmentation', 'DepthPlanar')
    else:
        _logger.info(f'Error: {seg_path} is not a valid segmentation path')
        return None
    
    depth_dir = os.path.splitext(depth_dir)[0]+'.lz4'
    if os.path.exists(depth_dir):
        return depth_dir
    else:
        _logger.error(f'Error: {depth_dir} does not exist')
        return None

def extra_process_gray_seg(gray_img, mask=None):
    # 因为天空是11，电线是18，先把天空改成19,这样膨胀腐蚀的时候不会把电线膨胀腐蚀
    gray_img[gray_img == 11] = 19
    # 然后膨胀腐蚀
    if mask is None:
        gray_img = erode(dilation(gray_img, 2), 1)
    else:
        gray_img = dilation_tiny_only(gray_img, mask)
    # 然后改回11
    gray_img[gray_img == 19] = 11
    return gray_img

def dilation_tiny_only(gray_seg,mask):
    '''只对细小物体de分割进行膨胀腐蚀'''
    res = gray_seg.copy()
    res[~mask] = 255
    # 此处核的参数为测试得出的最优参数
    res = erode(dilation(res, 2), 1)
    msk2 = res >=255
    res[msk2] = gray_seg[msk2]
    return res

def dilation(dpdata, d = 1):
    
    #expand = np.zeros_like(dpdata, dtype = np.float32)
    expand = dpdata.copy()
    
    # 膨胀特征
    d=1 + 2*d
    kernel = np.ones((d,d), np.uint8)
    # 实际操作是反向腐蚀，因为要让近处的深度覆盖远处的，即值小的优先
    #expand = cv2.dilate(expand, kernel, 1)
    expand = cv2.erode(expand, kernel, 1)   
        
    return expand

def erode(dpdata, d = 1):
    
    #expand = np.zeros_like(dpdata, dtype = np.float32)
    expand = dpdata.copy()
    
    # 膨胀特征
    d=1 + 2*d
    kernel = np.ones((d,d), np.uint8)
    # 实际操作是反向腐蚀，因为要让近处的深度覆盖远处的，即值小的优先
    expand = cv2.dilate(expand, kernel, 1)
    #expand = cv2.erode(expand, kernel, 1)   
        
    return expand

def _dilation_single_seg(seg_file):
    gray_img = _read_lz4(seg_file).copy()
    gray_img =extra_process_gray_seg(gray_img)
    _save_lz4(gray_img, seg_file)
    _logger.info(f'Finished dilation: {Path(seg_file).name}')

def tmp_dilate_seg(input_dir):
    if not Path(input_dir).exists():
        return
    seg_files = list(Path(input_dir).glob('*.lz4'))
    with Pool(cpu_count()) as p:
        p.map(partial(_dilation_single_seg), seg_files)    

def _process_segmentation_dir(seg_dir):
    seg_files = list(Path(seg_dir).iterdir())
    seg_files = [file for file in seg_files if file.is_file()]
    _logger.info(f'Found {len(seg_files)} segmentation images')
    folder = Path(seg_dir)
    gray_dir = folder / 'Graymap'
    color_dir = folder / 'Colormap'
    gray_dir.mkdir(exist_ok=True, parents=True)
    color_dir.mkdir(exist_ok=True, parents=True)
    exec_with_scale(partial(_process_segmentation_image,
                    color_dir=str(color_dir), gray_dir=str(gray_dir)), seg_files)


def _main(args):
    input_dir = args.input_dir
    assert Path(input_dir).exists(
    ), f'Error: input dir {input_dir} does not exist'
    _process_segmentation_dir(input_dir)

def main(input_dir):
    assert Path(input_dir).exists(
    ), f'Error: input dir {input_dir} does not exist'
    _logger.info(f'execute run_seg_gray_from_color.py ：Generate color image and graymap from segmentation images')
    _process_segmentation_dir(input_dir)
    

if __name__ == '__main__':
    # print('this is argument baseline pipeline')
    # parser = argparse.ArgumentParser(
    #     description=(
    #         'Generate color image and graymap from segmentation images.')
    # )
    # parser.add_argument('-i', '--input-dir', required=True,
    #                     help='Directory of the input images')
    # args = parser.parse_args()
    # _main(args)
    d=r"G:\Compressed\MH136p6_2x_Autel_left_2024-04-25-19-17-18\cube_rear\CubeSegmentation\1714043833499.webp"
    gray_d = r'G:\Compressed\MH136p6_2x_Autel_left_2024-04-25-19-17-18'
    color_d=gray_d
    _process_segmentation_image(Path(d),gray_d, color_d)
