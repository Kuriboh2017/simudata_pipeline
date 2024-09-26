#!/usr/bin/env python3
from functools import partial
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import lz4.frame as lz
import numpy as np
import pickle as pkl
import os
import cv2
import warnings

MAX_DEPTH = 1000
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


def _save_lz4(image_data, path):
    arr = np.ascontiguousarray(image_data)
    data = {
        'arr': lz.compress(arr, compression_level=3),
        'shape': image_data.shape,
        'dtype': image_data.dtype
    }
    with open(path, 'wb') as f:
        pkl.dump(data, f)


def get_thinmask_by_depth(depth_img):
    '''根据深度提取细小物体,depth_img必须为深度图'''
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to be treated as errors
        warnings.simplefilter("error")
        try:
            depth = depth_img.copy()
            far_value = 30  # 只关心30m以内的细小物体
            far_mask = depth > far_value
            threshold = 0.7  # 分辨细小物体边界的阈值，反复测试确定
            # 用于膨胀腐蚀的核，细小物体的定义为8个像素，故核的大小为9
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
            diff = np.abs(depth - closing)
            tiny_mask = diff > threshold
            tiny_mask[far_mask] = False
        except RuntimeWarning as e:
            _logger.error(f'Error processing {e}')
            return None
    return tiny_mask

def dilation_tiny_only(depth,mask):
    '''只对细小物体膨胀腐蚀，注意：只能用深度'''
    res = depth.copy()
    res[~mask] = 100
    # 此处核的参数为测试得出的最优参数
    res = erode(dilation(res, 2), 1)
    msk2 = res >=100
    res[msk2] = depth[msk2]
    return res

def _depth2disp(depth_file_dir, output_dir, baseline, dilation_tiny=True, save_disp=True):
    '''
    baseline双目基线长度，单位为m。
    本函数用于仿真ERP深度图转视差，已知仿真深度图使用float16存储。
    本函数要求输入深度图FoV为H360°xW180°。
    深度图中，非正深度和nan会被视为无效值且视差输出为nan，正无穷深度视差输出为0。
    '''
    depth_file_dir = Path(depth_file_dir)
    depth_img = _read_lz4(depth_file_dir).copy()
    # 对深度中细小物体的部分加粗
    if dilation_tiny:
        tiny_mask = get_thinmask_by_depth(depth_img)
        depth_img = dilation_tiny_only(depth_img,tiny_mask)
    
    h, w = depth_img.shape[:2]
    phi_l_start = 0.5 * np.pi - (0.5 * np.pi / w)
    phi_l_end = -0.5 * np.pi
    phi_l_step = np.pi / w
    phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
    phi_l_map = phi_l_range.reshape(1,-1).repeat(h, 0).astype(np.float32)

    nonpositive_mask = depth_img <= 0
    invalid_mask = nonpositive_mask | np.isnan(depth_img) # (depth_img > MAX_DEPTH)
    positive_inf_mask = ~nonpositive_mask & np.isinf(depth_img)
    
    depth_img[invalid_mask] = np.nan

    rho_r = np.sqrt(depth_img * depth_img + baseline * baseline - 
                    2 * depth_img * baseline * np.cos(phi_l_map + np.pi / 2))
    phi_r_map = np.arcsin(np.clip((depth_img * np.sin(phi_l_map) + baseline) / rho_r, -1, 1))
    disp = w * (phi_r_map - phi_l_map) / np.pi
    disp[disp < 0] = 0
    disp[positive_inf_mask] = 0
    
    #对视差加粗（改成对深度中的细小物体加粗，故视差不再加粗）
    # disp = disp.astype(np.float32)    # 只有float32才能进行膨胀腐蚀
    # disp = dilation(erode(disp, 2), 1)
    
    disp = disp.astype(np.float16)  # 为了减小存储空间，将视差存储为float16
    if (save_disp):
        _save_lz4(disp, output_dir / depth_file_dir.name)
    _logger.info(f'Finished depth-to-disparity conversion: {depth_file_dir.name}')
    return disp

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

def _dilation_disp(disp_file):
    disp = _read_lz4(disp_file).copy()
    disp = disp.astype(np.float32)
    disp = dilation(erode(disp, 2), 1)
    _save_lz4(disp, disp_file)
    _logger.info(f'Finished dilation: {Path(disp_file).name}')

def _create_output_dir(depth_dir, arg1):
    result = depth_dir.parent / arg1
    if result.exists():
        _logger.warning(f'Output dir {result} already exists')
    result.mkdir(parents=True, exist_ok=True)
    return result

def tmp_dilation_disp(input_dir):
    if not Path(input_dir).exists():
        return    
    disp_files = list(Path(input_dir).glob('*.lz4'))
    with Pool(cpu_count()) as p:
        p.map(partial(_dilation_disp), disp_files)

# def _main(args):
#     assert Path(args.input_dir).exists(
#     ), f'Error: depth dir {args.input_dir} does not exist'

#     up_baseline = args.up_baseline
#     down_baseline = args.down_baseline
#     depth_dir = Path(args.input_dir)
#     depth_files = list(depth_dir.glob('*.lz4'))
#     depth_files = sorted(depth_files)
    
#     output_dir_up = _create_output_dir(depth_dir, f'CubeDisparity_{str(up_baseline)}')
#     output_dir_down = _create_output_dir(depth_dir, f'CubeDisparity_{str(down_baseline)}')
#     with Pool(cpu_count()) as p:
#         p.map(partial(_depth2disp, output_dir=output_dir_up,
#               baseline=up_baseline), depth_files)

#     with Pool(cpu_count()) as p:
#         p.map(partial(_depth2disp, output_dir=output_dir_down,
#               baseline=down_baseline), depth_files)

def check_path_if_processed(input_dir,dp_dir):
    '''检查是否已经处理过'''
    if not Path(input_dir).exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return False
    else:
        disp_files = list(Path(input_dir).glob('*.lz4'))
        depth_files = list(Path(dp_dir).glob('*.lz4'))
        if len(disp_files) != len(depth_files):
            return False
        else:
            _logger.warning(f'Pass already processed {input_dir}')
            return True
        

def main(input_dir, up_baseline = 0.09, down_baseline=0.105, force_overlay = False):
    assert Path(input_dir).exists(
    ), f'Error: depth dir {input_dir} does not exist'
        
    _logger.info('execute run_paranoma_disparity_from_depth.py ：Generate disparity from depth')
    
    depth_dir = Path(input_dir)
    depth_files = list(depth_dir.glob('*.lz4'))
    depth_files = sorted(depth_files)
    
    # 生成上视视差图
    output_dir_up = _create_output_dir(depth_dir, f'CubeDisparity_{str(up_baseline)}')
    # 判断是否已经处理过，避免重复计算浪费时间
    if not check_path_if_processed(output_dir_up,depth_dir) or force_overlay:
        with Pool(cpu_count()) as p:
            p.map(partial(_depth2disp, output_dir=output_dir_up,
                baseline=up_baseline), depth_files)

    # 如果上下视基线相同，则不再生成下视视差图
    if up_baseline == down_baseline:
        return
    
    # 生成下视视差图
    output_dir_down = _create_output_dir(depth_dir, f'CubeDisparity_{str(down_baseline)}')    
    if not check_path_if_processed(output_dir_down,depth_dir) or force_overlay:
        with Pool(cpu_count()) as p:
            p.map(partial(_depth2disp, output_dir=output_dir_down,
                baseline=down_baseline), depth_files)
        
if __name__ == "__main__":
    _logger.info('this is /mnt/119-data/samba-share/simulation/code/lens_effects/batchfiles/run_panorama_disparity_from_depth.py')
    print('this is argument baseline pipeline')
    parser = argparse.ArgumentParser(
        description='Convert panorama depth to panorama disparity.')
    parser.add_argument('-i', '--input-dir', '--depth-dir', required=True,
                        help='path of the input image')
    parser.add_argument('-ub', '--up-baseline', type=float, default=0.09,
                        help='up camera baseline')
    parser.add_argument('-db', '--down-baseline', type=float, default=0.105,
                        help='down camera baseline')
    args = parser.parse_args()
    _main(args)
