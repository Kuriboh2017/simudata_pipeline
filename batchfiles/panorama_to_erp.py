#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
from PIL import Image
import lz4.frame as lz
import numpy as np
import os
import pickle as pkl
from scipy import interpolate

from resize_min_max_bilinear_class import Resize
from functools import partial
from multiprocessing import Pool, cpu_count
import glob
import shutil
resize = Resize()


def _fill_nan_with_border_values(arr):
    non_nan_indices = np.argwhere(~np.isnan(arr))
    non_nan_values = arr[~np.isnan(arr)]
    nan_indices = np.argwhere(np.isnan(arr))
    filled_nan_values = interpolate.griddata(
        non_nan_indices, non_nan_values, nan_indices, method='nearest')
    arr[np.isnan(arr)] = filled_nan_values
    return arr


def _replace_boundary(arr):
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)
    arr[max_indices] = max_val - 1
    return arr

def _customized_remap(input_img, col_data, row_data, func):
    col_data = _fill_nan_with_border_values(col_data.copy())
    row_data = _fill_nan_with_border_values(row_data.copy())
    col_data_lbound = np.floor(col_data).astype(int)
    col_data_ubound = np.ceil(col_data).astype(int)
    row_data_lbound = np.floor(row_data).astype(int)
    row_data_ubound = np.ceil(row_data).astype(int)

    # Handle out of bounds (e.g. image boundary 1280, 2560) indices
    col_data_lbound = _replace_boundary(col_data_lbound.copy())
    row_data_lbound = _replace_boundary(row_data_lbound.copy())
    col_data_ubound = _replace_boundary(col_data_ubound.copy())
    row_data_ubound = _replace_boundary(row_data_ubound.copy())
    depth_left_top = input_img[row_data_lbound, col_data_lbound]
    depth_left_bot = input_img[row_data_ubound, col_data_lbound]
    depth_right_top = input_img[row_data_lbound, col_data_ubound]
    depth_right_bot = input_img[row_data_ubound, col_data_ubound]
    depth_grid_in_channel = np.stack([
        depth_left_top, depth_left_bot, depth_right_top, depth_right_bot
    ])
    return func(depth_grid_in_channel, axis=0)

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

def relabel_seg(segfile):
    seg = segfile.copy()
    
    # 假如已经relabel了，就跳过
    tmp = seg.reshape(-1,1)
    indexes =np.unique(tmp, axis=0)
    if 1 in indexes:
        return seg
    
    # seg: 0物体，11天空，18电线 -> 0天空，1其他，2电线
    other_mask = seg == 0
    sky_mask = seg == 11
    wire_mask = seg == 18
    
    seg[sky_mask] = 0
    seg[other_mask] = 1
    seg[wire_mask] = 2
    return seg

def crop_erp(data, is_disp, is_seg_gray, data_dir):
    #crop_fov = (185, 120)  # 训练dataloader输出的最终fov
    #crop_shape = (1408, 1280)
    
    crop_fov = (360, 120)  # 训练dataloader输出的最终fov
    crop_shape = (2740, 1280)
    
    # 跳过已处理和错误shape的图片
    if data is None:
        print(f'error: {data_dir} is None, will remove in check stage')
        return data
    if data.shape[:2] != (5120, 2560):
        if data.shape[:2] == crop_shape:
            print(f'warning: {data_dir} already resized, skip resizing')
        else:
            print(f'error: {data_dir} shape is {data.shape}, not {crop_shape}')
        return data
    
    # resize
    resize_h = round(crop_shape[0] / crop_fov[0] * 360)
    resize_w = round(crop_shape[1] / crop_fov[1] * 180)
    dsize = (resize_w, resize_h)
    
    if is_disp:       
        resize_scale = resize_w / data.shape[1]    # 视差跟尺寸也有关，所以视差图resize之后要等比例缩放
        data = resize.resize(data, resize_h, resize_w, interpolation='max', ipsize=2)
        data = (data * resize_scale).astype(np.float16)  ## 除了某些计算过程（比如膨胀腐蚀）必要float32，改回float16节省空间   
    elif is_seg_gray:        
        data = relabel_seg(data)
        data = resize.resize(data, resize_h, resize_w, interpolation='max', ipsize=2)
        data = data.astype(np.uint8)  
    else:
        data = cv2.resize(data, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    crop_w = crop_shape[1]
    rescale_w = data.shape[1]
    dy = int(round((rescale_w-crop_w)/2))+1
    data = data[:, dy:dy+crop_w]  # (2740, 1920, 3) -> (2740, 1280, 3)

    # up_data = np.vstack([data[-19:, :], data[:(1370+19), :]])
    # down_data = np.vstack([data[(1370-19):, :], data[:19, :]])
    # return down_data, up_data, data
    return data


def append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


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


def resize_all(input_path, output_path=None):
    '''
    input_path=输入数据集,例如xx/2x_MX128_bigCity_2024-05-15-17-23-16
    output_path=输出路径文件夹,例如xx/2x_MX128_bigCity_2024-05-15-17-23-16_out
    '''
    if not os.path.exists(input_path):
        return
    
    # 检查文件夹是否已经处理过了
    if os.path.exists(input_path + '/resize_done.log'):
        return
    
    # 获取cube_x目录下的全部.lz4,.webp,.png文件    
    input_files = [str(p) for p in Path(input_path).rglob('*.lz4') if p.is_file() and 'cube' in str(p)]
    input_files.extend([str(p) for p in Path(input_path).rglob('*.webp') if p.is_file() and 'cube' in str(p)])
    input_files.extend([str(p) for p in Path(input_path).rglob('*.png') if p.is_file() and 'cube' in str(p)])
    #input_files = list(Path(input_path).iterdir())
    #assert input_files, 'Error: can\'t find the any image files!'
    print(len(input_files))
    if len(input_files) == 0:
        print(f'{input_path} no panorama images, no need to resize')
        os.mkdir(input_path + '/resize_done.log') 
        return
    
    input_files = np.sort(np.array(input_files))
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        print(f'resize output_path = {Path(output_path).absolute()}')
    # for f in input_files:
    #     _process_image(f)
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image,output_dir=output_path), input_files)
        
    # 处理完毕
    os.mkdir(input_path + '/resize_done.log')    

def _process_image(input_file, output_dir=None):
    '''
    input_file: 文件全路径
    output_dir: 输出文件夹
    '''
    input_file = str(input_file)

    if '_mask.' in input_file:  # 过滤特定的mask文件，例如机身mask
        return 
    
    try:  
        if input_file.endswith('.lz4'):
            #print(f'read {input_file}')
            input_img = _read_lz4(input_file)
        else:
            #print(f'read {input_file}')
            input_img = cv2.imread(input_file)      # 用cv2读取而不是Image,方便跟np数组转换
            if input_img is None:
                print(f'error: {input_img} is None, will remove in check stage')
                return
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)  # cv2默认是bgr，转成rgb
    except EOFError:
        print(f"ERROR:No data to load from {input_file}.")
        os.remove(input_file)
        return
        
    is_disp = False
    is_seg_gray = False
    if 'Disparity' in str(input_file):
        is_disp = True        
    elif 'Segmentation' in str(input_file):        
        is_seg_gray = True

    #print('crop_erp !!!!!!')
    copy_img = input_img.copy() # 原始数据有的是只读的
    inter_img = crop_erp(copy_img, is_disp, is_seg_gray, input_file)
    
    #print('crop_erp done !!!!!!')
    if output_dir is None:
        inter_dir = input_file  # 默认覆盖原文件
    else:
        inter_dir = Path(output_dir) / Path(input_file).name
        
    if str(inter_dir).endswith('.lz4'):
        _save_lz4(inter_img, inter_dir)
    else:
        #cv2.imwrite(str(inter_dir), np.uint8(inter_img))
        Image.fromarray(inter_img).save(inter_dir, lossless = False)    # 用Image有损压缩

    print(f'saved resized image to {inter_dir}')

def replacefolder(src):
    if 'inter' not in src:
        return
    if not os.path.exists(src):
        return
    dst = str(Path(src).parent).replace('_inter', '')
    shutil.rmtree(src.replace('_inter', ''))
    shutil.move(src, dst)
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Rectify 1 fisheye image to a erp image.')
    # parser.add_argument('-i', '--input-image-path', required=True,
    #                     help='path of the input image')
    # parser.add_argument('-o0', '--output-image-path0',
    #                     help='path 0 of the output image')
    # parser.add_argument('-o1', '--output-image-path1',
    #                     help='path 1 of the output image')
    # parser.add_argument('-d', '--disparity', action='store_true',
    #                     help='whether or not processing disparity image')
    # parser.add_argument('-g', '--segmentation-graymap', action='store_true',
    #                     help='whether or not processing segmentation grapmap image')
    # args = parser.parse_args()
    # _main(args)

    # d = '/mnt/119-data/samba-share/simulation/train/whitewall/2x_whitewall_2024-03-16-03-45-55/cube_front/CubeScene/1710468164532590080.webp'
    # output_dir = '/mnt/119-data/samba-share/simulation/train/whitewall/'
    # os.makedirs(output_dir, exist_ok=True)
    # _process_image(d, output_dir)
    d='/mnt/119-data/samba-share/simulation/train_out/2x_1007/2x_Autel2_battleground_kit/2x_Autel2_battleground_kit_2023-10-07-13-52-39'
    resize_all(d)
    folders = [str(p) for p in Path(d).rglob('cube_front') if p.is_dir()]
    print(len(folders))
    # d = r"G:\1715686792268_erp.webp"
    # file = cv2.imread(d)
    # file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    # file = crop_erp(file, False, False, d)
    # Image.fromarray(file).save(d.replace('.webp', '_erp.webp'), lossless = False)