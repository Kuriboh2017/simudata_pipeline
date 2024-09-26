#!/usr/bin/env python3
import argparse
import distutils.spawn
import logging
import shutil
import glob
import os

from pathlib import Path
from subprocess import run

import run_seg_gray_from_color
import run_panorama_disparity_from_depth
import run_pinholes_disparity_from_depth
import simu_params  # 用模拟参数代替cmd传参，实现调用可控

import time

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

kRemappingTablesDir = '/mnt/119-data/samba-share/simulation/code/2x'
#kRemappingTablesDir = r"G:\simu3\lens_effects\test_2x"
kDownsampledRootDir = '/mnt/119-data/R23024/data/junwu/data/synthesis'
kDownsampledRootDir_erp = '/mnt/119-data/R22612/Data/ERP/train/synthesis'

# 默认上下视baseline
up_baseline = 0.09
down_baseline = 0.105
pinhole_baseline = 0.06
focallength = 1446.238224784178
test_var = 0

def _run_exe(exe, input_dir, args):
    generate_disparity_exe = distutils.spawn.find_executable(exe)
    assert generate_disparity_exe, 'Error: executable `{exe}` is not available!'
    cmds = [generate_disparity_exe, f'--input-dir={input_dir}']
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _run_pre_fisheye_steps(input_dir):
    '''
    Step 1: Preprocess: generate panorama disparity and segmentation graymap
      by running `run_panorama_disparity_from_depth.py` and 
      `run_seg_gray_from_color.py`
    '''
    input_dir = Path(input_dir)
    depth_dir = input_dir / 'cube_front' / 'CubeDepth'
    segment_dir = input_dir / 'cube_front' / 'CubeSegmentation'

    if segment_dir.exists():
        run_seg_gray_from_color.main(str(segment_dir))
    
    # ====== Generate Panorama Disparity ======
    if depth_dir.exists():
        # 默认0.09上视， 0.105下视
        if 'MX128' in str(input_dir):
            up_baseline = 0.128
            down_baseline = 0.105
            run_panorama_disparity_from_depth.main(str(depth_dir),up_baseline=up_baseline)
        elif 'MH136p6' in str(input_dir):
            up_baseline = 0.1366
            down_baseline = 0.1366
            run_panorama_disparity_from_depth.main(str(depth_dir),up_baseline=up_baseline, down_baseline=down_baseline)
        else:
            up_baseline = 0.09
            down_baseline = 0.105
            run_panorama_disparity_from_depth.main(str(depth_dir))
    # ====== Generate Panorama Disparity ======

    ph_dp_dir = [str(p) for p in input_dir.rglob("DepthPlanar") if p.is_dir()] #input_dir / 'left' / 'DepthPlanar'
    #ph_seg_dir = input_dir / 'left' / 'Segmentation'
    for ph_dir in ph_dp_dir:
        if not Path(ph_dir).exists():
            return
        if 'front_left' in ph_dir:
            pinhole_baseline = 0.06
            focallength = 834.0642386183716
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=pinhole_baseline, focal_length=focallength)
        elif 'up_left' in ph_dir:
            pinhole_baseline = 0.04
            focallength = 662.7394008259646
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=pinhole_baseline, focal_length=focallength)
        elif 'down_left' in ph_dir:
            pinhole_baseline = 0.08
            focallength = 662.7394008259646
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=pinhole_baseline, focal_length=focallength)
        else:
            pinhole_baseline = 0.06
            focallength = 1446.238224784178
            run_pinholes_disparity_from_depth.main(ph_dir)
            
        run_seg_gray_from_color.main(ph_dir.replace("DepthPlanar",'Segmentation'))


def _delete_cubedisparity_subdirs(parent_dir, prefix="CubeDisparity_"):
    '''
    删除parent_dir目录下CubeDisparity_子目录
    parent_dir：通常是xx/cube_front
    '''
    parent_dir_path = Path(parent_dir)
    if not parent_dir_path.exists():
        _logger.error(f"Error: The directory {parent_dir} does not exist.")
        return
    if not parent_dir_path.is_dir():
        _logger.error(f"Error: {parent_dir} is not a directory.")
        return
    for subdir in parent_dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith(prefix):
            try:                
                shutil.rmtree(subdir)
                _logger.info(f"Deleted {subdir}")
            except Exception as e:
                _logger.info(f"Error deleting {subdir}: {e}")


def move_fisheye(input_dir):
    from distutils.dir_util import copy_tree
    # 拷贝文件夹到新目录
    out_dir = input_dir.replace('train','train_out', 1)
    # 检查是否已经拷贝过了
    input_disp_ph_dir = Path(input_dir) / 'left' / 'Disparity'
    input_disp_cube_dir = Path(input_dir) / 'cube_front' / f'CubeDisparity_{str(up_baseline)}'
    if not input_disp_ph_dir.exists() and not input_disp_cube_dir.exists() and Path(out_dir).exists():
        _logger.warning(f"Warning: {input_dir} has been moved yet.")
        return out_dir
    
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)
    # 除了深度图，其他都拷贝input_dir到out_dir
    copy_tree(input_dir, out_dir)
    shutil.rmtree(Path(out_dir) / 'cube_front'/ 'CubeDepth', ignore_errors=True)
    shutil.rmtree(Path(out_dir) / 'left' / 'DepthPlanar', ignore_errors=True)
    # 拷贝完成后删除原文件夹的视差图
    #_delete_cubedisparity_subdirs(Path(input_dir) / 'cube_front')
    shutil.rmtree(Path(input_dir) / 'cube_front' / f'CubeDisparity_{str(up_baseline)}', ignore_errors=True)
    shutil.rmtree(Path(input_dir) / 'cube_front' / f'CubeDisparity_{str(down_baseline)}', ignore_errors=True)
    shutil.rmtree(Path(input_dir) / 'left' / 'Disparity', ignore_errors=True)
    
    return out_dir

def _resize_fisheye(input_dir):
    '''
    input_dir: 拷贝后的文件夹,通常是train_out/XX/2x_scenename_time
    '''
    from panorama_to_erp import resize_all
    
    # 开始resize
    _logger.info(f'_resize_fisheye::START resize fisheye: out_dir={input_dir}')
    resize_all(input_dir)    
    _logger.info(f'_resize_fisheye::resize fisheye done! out_dir={input_dir}')
    
    return input_dir

def _check_fisheye(input_dir):
    '''
    文件自检
    '''
    import erp_gt_check_cv
    erp_gt_check_cv.multiprocess_compare_after_resize(input_dir, down_baseline, pinhole_baseline, focallength)
    _logger.info(f'_check_fisheye::{input_dir} check fisheye done!')
    
def get_last_modified_time(file_path):
    ''' 获取文件的最后修改时间'''
    last_modified_time = os.path.getmtime(file_path)
    
    # 将时间戳转换为可读格式
    formatted_time = time.ctime(last_modified_time)
    
    return formatted_time 

# 从整个大文件夹筛选（递归方式）出需要处理的文件夹，然后处理
def process(folder, filter_str='imu'):
    '''
    filter_str:默认选imu而不是cube_front或left，因为有的数据集并不是都包含这两者，防止漏筛
    '''
    d= folder
    # 获取待处理的文件夹
    folders = [str(p) for p in Path(d).rglob(filter_str) if p.is_dir()]
    print(len(folders))
    
    n=0
    for f in folders:
        f = str(Path(f).parent)
        if os.path.exists(f):
            n += 1
            '''
            # Step 1: pre_fisheye, disparity, segmentation
            _logger.info(f'{n}:{len(folders)} processing {f}')
            _run_pre_fisheye_steps(f)
            _logger.info('Step1: pre_fisheye_steps done!')
            '''
            # Step 2: resize fisheye
            out_dir = move_fisheye(f)
            out_dir = _resize_fisheye(out_dir)
            out_dir = f.replace('train','train_out', 1)
            # Step 3: check fisheye
            _check_fisheye(out_dir)
            # Step 4: genarate filelist
            
            
            
            
def rm_disparity(folder):
    '''删除视差图'''
    folders = [str(p) for p in Path(folder).rglob('cube_front') if p.is_dir()]
    for f in folders:
        
        _delete_cubedisparity_subdirs(f)
            
if __name__ == '__main__':   
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1007"
    # rm_disparity(d)
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1017"
    # rm_disparity(d)r
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1024"
    # rm_disparity(d)    
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1116"
    # rm_disparity(d) /mnt/119-data/samba-share/simulation/train_out/whitewall      

    #d='/mnt/119-data/samba-share/simulation/train/2x_1017/2x_Autel_MediterraneanIsle/'
    
    # disp = [str(p) for p in Path(d).rglob("Disparity") if p.is_dir()]
    # for f in disp:
    #     shutil.rmtree(f)
    #     print(f'Deleted {f}')
    # d='/mnt/119-data/samba-share/simulation/train/2x_1007'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/2x_1017'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/2x_1024'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/2x_1116'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/2x_lowfly'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/2x_substation_1208'
    # print(d)
    # process(d,'imu')
    d='/mnt/119-data/samba-share/simulation/train/evo2_pinhole'
    print(d)
    process(d,'imu')

    # d='/mnt/119-data/samba-share/simulation/train/hypertiny_0102'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/hypertiny_0105'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/hypertiny_0130'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/hypertiny_test'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/MH136p6'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/MX128'
    # print(d)
    # process(d,'imu')
    
    # d='/mnt/119-data/samba-share/simulation/train/MX128_hypertiny_0204'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/shaoshan'
    # print(d)
    # process(d,'imu')
    # d='/mnt/119-data/samba-share/simulation/train/whitewall'
    # print(d)
    # process(d,'imu')    