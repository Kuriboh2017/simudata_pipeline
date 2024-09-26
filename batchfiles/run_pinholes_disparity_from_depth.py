#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import re
import json
import numpy as np
import os
import lz4.frame as lz
import pickle as pkl
import simu_params
from run_panorama_disparity_from_depth import get_thinmask_by_depth, dilation_tiny_only, check_path_if_processed
import glob
kCurrentDir = os.path.dirname(os.path.abspath(__file__))


def _read_pfm(file):
    """ Read a pfm file """
    print('warning::pfm is not used anymore')
    return 


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


def _depth2disp(depth_file, output_dir, args, dilation_tiny=False, save_disp=False):
    depth_file = Path(depth_file)
    if depth_file.suffix == '.lz4':
        depth_data = _read_lz4(depth_file)
    elif depth_file.suffix == '.npz':
        depth_data = np.load(str(depth_file))['arr_0']
    elif depth_file.suffix == '.pfm':
        depth_data, _ = _read_pfm(depth_file)
    else:
        raise NotImplementedError

    depth_img = _read_lz4(depth_file).copy()
    # 对深度中细小物体的部分加粗
    if dilation_tiny:
        tiny_mask = get_thinmask_by_depth(depth_img)
        depth_data = dilation_tiny_only(depth_img,tiny_mask)

    baseline = args.baseline
    focal_length = args.focal_length
    
    disparity = baseline * focal_length / depth_data
    disparity = disparity.astype(np.float16)

    if save_disp:
        output_path = output_dir / depth_file.name
        if output_path.suffix in ('.lz4', '.pfm'):
            output_path = output_path.with_suffix('.lz4')
            _save_lz4(disparity, output_path)
        elif output_path.suffix == '.npz':
            np.savez_compressed(output_path, arr_0=disparity)
        else:
            raise NotImplementedError
    print(f'Processed depth-to-disparity using file: {depth_file.name}')
    return disparity


def _main(args):
    assert Path(args.input_dir).exists(
    ), f'Error: depth dir {args.input_dir} does not exist'

    depth_dir = Path(args.input_dir)
    depth_files = list(depth_dir.glob('*.npz'))
    depth_files.extend(list(depth_dir.glob('*.lz4')))
    depth_files.extend(list(depth_dir.glob('*.pfm')))
    depth_files = sorted(depth_files)
    output_dir = args.output_dir
    if output_dir is None or output_dir == 'None':
        output_dir = depth_dir.parent / 'Disparity'
    else:
        output_dir = Path(args.output_dir)
    assert not output_dir.exists(
    ), f'Error: output dir {output_dir} already exists'
    output_dir.mkdir(parents=True)
    with open(output_dir / 'camera_parameters.json', 'w') as f:
        cam_params = {'baseline_in_meters': args.baseline,
                      'focal_length_in_pixels': args.focal_length}
        json.dump(cam_params, f)
        print(
            f'baseline: {args.baseline} meters; focal length: {args.focal_length} pixels')

    with Pool(cpu_count()) as p:
        p.map(partial(_depth2disp, output_dir=output_dir,
              args=args), depth_files)

def main(input_dir, baseline = 0.06, focal_length=1446.238224784178):
    assert Path(input_dir).exists(
    ), f'Error: depth dir {input_dir} does not exist'

    depth_dir = Path(input_dir)
    depth_files = list(depth_dir.glob('*.npz'))
    depth_files.extend(list(depth_dir.glob('*.lz4')))
    depth_files.extend(list(depth_dir.glob('*.pfm')))
    depth_files = sorted(depth_files)

    output_dir = depth_dir.parent / 'Disparity'
    args = simu_params.ParaPinhole(focal_length, baseline)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # 判断是否已处理
    if check_path_if_processed(output_dir, depth_dir):
        return
    with Pool(cpu_count()) as p:
        p.map(partial(_depth2disp, output_dir=output_dir,
              args=args), depth_files)    

def fix_disp_good():

    # 获取数据集
    dir = '/mnt/119-data/samba-share/simulation/train/'
    folders = glob.glob(dir + '/*')
    
    filter = ['2x_1007', '2x_1017', '2x_1024', '2x_1116', 'hypertiny_0102', ]   # 已知错误但是还没有合并的
    for f in folders:
        if Path(f).name not in filter:
            continue
        print(f)
        disp_folder = [str(p) for p in Path(f).rglob('Disparity') if p.is_dir()]
        
        #获取单个数据集
        for f2 in disp_folder:
            print(f'Checking {f2}')
            bDispGood=True
            disp_dir = list(Path(f2).rglob('*.lz4'))
            # 多进程处理
            with Pool(cpu_count()) as p:
                p.map(partial(compare_disp_good), disp_dir)

        

            

def compare_disp_good(disp_dir):
    disp = _read_lz4(disp_dir)
    depth_dir = str(disp_dir).replace('Disparity','DepthPlanar')

    
    # 重建视差
    depth_img = _read_lz4(depth_dir).copy()
    # 对深度中细小物体的部分加粗
    tiny_mask = get_thinmask_by_depth(depth_img)
    depth_data = dilation_tiny_only(depth_img,tiny_mask)

    baseline = 0.06
    focal_length = 1446.238224784178
    disparity = baseline * focal_length / depth_data
    disparity = disparity.astype(np.float16)
    # 覆盖原视差
    _save_lz4(disparity, disp_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Convert 1 panorama image to the 3-pinholes directly.')
    # parser.add_argument('-i', '--input-dir', '--depth-dir', required=True,
    #                     help='path of the input image')
    # parser.add_argument('-o', '--output-dir', '--disparity-dir',
    #                     help='path of the output disparity image')
    # parser.add_argument('-b', '--baseline', type=float, default=0.06,
    #                     help='baseline of the stereo camera')
    # parser.add_argument('-f', '--focal-length', type=float, default=1446.238224784178,
    #                     help='focal length of the stereo camera')
    # args = parser.parse_args()
    d = '/mnt/119-data/samba-share/simulation/tmp/2x_MX128_Autel3_RuralAustralia_forest_2024-01-22-21-44-04copy/left/DepthPlanar/1705931037962691584.lz4'
    fix_disp_good()
    #_main(args)


