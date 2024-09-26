#!/usr/bin/env python3
# Do the following:
# cube_front: down -> group0/cam0_0
# cube_front: up -> group1/cam1_1
# cube_rear: up -> group1/cam1_0
# cube_below: down -> group0/cam0_1

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import cv2
import itertools
import logging
import lz4.frame as lz
import math
import numpy as np
import os
import pickle as pkl
import shutil
import uuid

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

kCurrentDir = os.path.dirname(os.path.abspath(__file__))

DEPTH = 'Depth'


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
        print(f"Directory '{src}' has been copied to '{dest}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except shutil.Error as e:
        print(f"Error: {e}")


def _copy_file(file, src_dir, dest_dir):
    relative_path = file.relative_to(src_dir)
    assert len(
        relative_path.parts) >= 2, f'Error: len(relative_path.parts) = {len(relative_path.parts)}'
    if relative_path.parts[0] in ('CubeScene_3to1'):
        relative_path = Path('Image_3to1').joinpath(*relative_path.parts[1:])
    elif relative_path.parts[0] in ('CubeScene_erp'):
        relative_path = Path('Image_erp').joinpath(*relative_path.parts[1:])
    elif relative_path.parts[0] in ('CubeSegmentation'):
        relative_path = Path('Segmentation').joinpath(*relative_path.parts[1:])
    elif relative_path.parts[0] in ('CubeDepth_3to1', 'CubeDepth'):
        relative_path = Path(f'{DEPTH}').joinpath(*relative_path.parts[1:])
    elif relative_path.parts[0] in ('CubeDisparity_0.09_erp'):
        relative_path = Path('Disparity_0.09_erp').joinpath(
            *relative_path.parts[1:])
    elif relative_path.parts[0] in ('CubeDisparity_0.105_erp'):
        relative_path = Path('Disparity_0.105_erp').joinpath(
            *relative_path.parts[1:])
    (dest_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy(str(file), str(dest_dir / relative_path))
    print(f'Copied {file} to {dest_dir / relative_path}')


def _copy_files_if_with_suffix(src_dir, dest_dir, suffix):
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = src_dir.rglob('*')
    filtered_files = [file for file in files if file.is_file()
                      and suffix in file.stem]
    with Pool(cpu_count()) as p:
        p.map(partial(_copy_file, src_dir=src_dir,
              dest_dir=dest_dir), filtered_files)


def _rename_file(file, suffix):
    new_name = file.stem[:-len(suffix)] + file.suffix
    file.rename(file.parent / new_name)
    print(f'Renamed {file} to {file.parent / new_name}')


def _rename_files_recursive(directory: Path, suffix: str):
    files = directory.rglob(f'*{suffix}*')
    filtered_files = [file for file in files if file.is_file()
                      and file.stem.endswith(suffix)]
    with Pool(cpu_count()) as p:
        p.map(partial(_rename_file, suffix=suffix), filtered_files)


def _rotate_rgb(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.rotate(img, cv2.ROTATE_180)
    rot_img_path = img_path.parent / f'rot_180_{img_path.name}'
    cv2.imwrite(str(rot_img_path), img)
    os.replace(rot_img_path, img_path)
    _logger.info(f'Rotated {img_path}')


def _rotate_npz(img_path):
    _logger.info(f'Rotating {img_path}')
    depth = np.load(img_path, allow_pickle=True)['arr_0']
    depth = np.flip(np.flip(depth, axis=0), axis=1)
    rot_img_path = img_path.parent / f'rot_180_{img_path.name}'
    np.savez_compressed(rot_img_path, depth)
    os.replace(rot_img_path, img_path)
    _logger.info(f'Rotated {img_path}')


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


def _rotate_lz4(img_path):
    data = _read_lz4(img_path)
    data = np.flip(np.flip(data, axis=0), axis=1)
    rot_img_path = img_path.parent / f'rot_180_{img_path.name}'
    _save_lz4(data, rot_img_path)
    os.replace(rot_img_path, img_path)
    _logger.info(f'Rotated {img_path}')


def _run_parallel(func, files):
    with Pool(cpu_count()) as p:
        p.map(func, files)


def _swap_directory_names(path1: Path, path2: Path):
    if not path1.is_dir() or not path2.is_dir():
        raise ValueError("Both paths should point to directories.")
    tmp_dir = path1.parent / str(uuid.uuid4())
    path1.rename(tmp_dir)
    path2.rename(path1)
    tmp_dir.rename(path2)
    _logger.info(f'Swapped {path1} and {path2}')


def _process_up_fisheye_dir(up_fisheye_dir):
    assert up_fisheye_dir.exists(
    ), f'Input directory does not exist: {up_fisheye_dir}'
    # Rotate `up-fisheye` images 180 degrees
    files = up_fisheye_dir.rglob('*')
    img_files = [file for file in files if file.is_file()]
    rgb_files = [file for file in img_files if file.name.endswith('.webp')]
    npz_files = [file for file in img_files if file.name.endswith('.npz')]
    lz4_files = [file for file in img_files if file.name.endswith('.lz4')]
    if rgb_files:
        _run_parallel(_rotate_rgb, rgb_files)
    if npz_files:
        _run_parallel(_rotate_npz, npz_files)
    if lz4_files:
        _run_parallel(_rotate_lz4, lz4_files)

    # Swap left and right images
    cam0_dir = up_fisheye_dir / 'cam1_0'
    cam1_dir = up_fisheye_dir / 'cam1_1'
    if cam0_dir.exists() and cam1_dir.exists():
        _swap_directory_names(cam0_dir, cam1_dir)


def _depth2disp(depth_file, recti_params, baseline):
    if depth_file.suffix == '.lz4':
        depth_data = _read_lz4(depth_file)
    else:
        depth_data = np.load(str(depth_file))['arr_0']
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    zDatas = np.zeros([ori_input_h, ori_input_w], dtype=np.float32)
    partZDatas = np.zeros([avg_input_h, ori_input_w], dtype=np.float32)
    for k in range(3):
        k_start = k * avg_input_h
        k_end = k_start + avg_input_h
        partDepth = depth_data[k_start:k_end, :]
        p = recti_params[k].param
        for j, i in itertools.product(range(avg_input_h), range(ori_input_w)):
            x = math.pow((i - p[2]), 2) / math.pow(p[0], 2)
            y = math.pow((j - p[3]), 2) / math.pow(p[1], 2)
            zData = math.sqrt(math.pow(partDepth[j, i], 2) / (x + y + 1))
            partZDatas[j, i] = zData
        zDatas[k_start:k_end, :] = partZDatas
    zDatas = baseline * p[0] / zDatas
    zDatas = zDatas.astype(np.float16)
    # Save the disparity image to the same file as the depth image
    disparity_img_path = depth_file.parent / f'disparity_{depth_file.name}'
    if depth_file.suffix == '.lz4':
        _save_lz4(zDatas, disparity_img_path)
    else:
        np.savez_compressed(disparity_img_path, arr_0=zDatas)
    os.replace(disparity_img_path, depth_file)
    _logger.info(f'Processed depth-to-disparity using file: {depth_file.name}')


class _RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


def read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
    right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6
    multi_recti_params = []
    multi_recti_params_node = fs.getNode("multiRectis")
    for i in range(multi_recti_params_node.size()):
        n = multi_recti_params_node.at(i)
        tmp = _RectiSpec(n.at(3).real(), n.at(2).real(), [n.at(4).real(), n.at(
            5).real(), n.at(6).real(), n.at(7).real()], n.at(8).real())
        multi_recti_params.append(tmp)

    translation = fs.getNode("trans").mat().reshape(-1) * 0.01
    rotation = fs.getNode("rmat").mat()
    return left_intri, right_intri, translation, rotation, multi_recti_params


def _convert_depth_to_disparity(dir, baseline, args):
    rectify_config = args.rectify_config
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"
    if not os.path.exists(rectify_config):
        assert False, f'Error: rectification config file {rectify_config} does not exist'
    l, r, trans, rmat, multispecs = read_yaml(rectify_config)

    # Depth or depth
    files = list(dir.rglob('*'))
    print(f'Found {len(files)} files in {dir}')
    depth_files = [file for file in files if file.is_file()
                   and file.suffix in ('.lz4', '.npz')]
    print(f'Found {len(depth_files)} depth files in {dir}')
    with Pool(cpu_count()) as p:
        p.map(partial(_depth2disp, recti_params=multispecs,
              baseline=baseline), depth_files)
    dir.rename(dir.parent / 'Disparity_3to1')


def _handle_erp_disparity_folders(down_dir, up_dir):
    down_disparity_erp = down_dir / 'Disparity_0.105_erp'
    up_disparity_erp = up_dir / 'Disparity_0.105_erp'
    down_disparity_erp.rename(down_dir / 'Disparity_erp')
    up_disparity_erp.rename(up_dir / 'Disparity_erp')
    # Remove the disparity folders with the wrong baseline
    shutil.rmtree(down_dir / 'Disparity_0.105_erp')
    shutil.rmtree(up_dir / 'Disparity_0.105_erp')


def _process_dir(output_dir):
    _logger.info(
        'Renaming the folders to match the real-world drone data folder structure.')
    cube_front_dir = Path(output_dir) / 'cube_front'
    cube_rear_dir = Path(output_dir) / 'cube_rear'
    # cube_below_dir = Path(output_dir) / 'cube_below'
    group0_0_dir = Path(output_dir) / 'group0' / 'cam0_0'
    group0_1_dir = Path(output_dir) / 'group0' / 'cam0_1'
    group1_0_dir = Path(output_dir) / 'group1' / 'cam1_0'
    group1_1_dir = Path(output_dir) / 'group1' / 'cam1_1'
    _copy_files_if_with_suffix(cube_front_dir, group0_0_dir, '_down')
    _copy_files_if_with_suffix(cube_front_dir, group1_1_dir, '_up')
    _copy_files_if_with_suffix(cube_rear_dir, group1_0_dir, '_up')
    _copy_files_if_with_suffix(cube_rear_dir, group0_1_dir, '_down')
    _rename_files_recursive(group0_0_dir, '_down')
    _rename_files_recursive(group1_1_dir, '_up')
    _rename_files_recursive(group1_0_dir, '_up')
    _rename_files_recursive(group0_1_dir, '_down')
    shutil.rmtree(cube_front_dir)
    shutil.rmtree(cube_rear_dir)
    # shutil.rmtree(cube_below_dir)
    _logger.info('Done renaming!')

    _logger.info(
        'Rotating the up fisheye images 180 degrees to let left images has depth and segmentation data.')
    up_fisheye_dir = Path(output_dir) / 'group1'
    _process_up_fisheye_dir(up_fisheye_dir)
    _logger.info('Done rotation!')

    _logger.info('Converting depth to disparity')
    # Bottom camera baseline: 0.105
    _convert_depth_to_disparity(group0_0_dir / f'{DEPTH}', 0.105, args)
    # Up camera baseline: 0.105
    _convert_depth_to_disparity(group1_0_dir / f'{DEPTH}', 0.105, args)
    # Handle ERP disparity folders
    _handle_erp_disparity_folders(group0_0_dir, group1_0_dir)
    _logger.info('Done conversion!')


def _main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or f'{Path(input_dir)}_renamed'
    copy_directory(input_dir, output_dir)
    _process_dir(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('After running fisheye conversion, '
                     '1. Rename the 3 cube folders to match the real-world '
                     'drone data folder structure.'
                     '2. Rotate the up fisheye images 180 degrees to '
                     'let left images has depth and segmentation data.'
                     '3. Convert depth to disparity with different baselines'
                     )
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir',
                        help='Directory of the output images')
    parser.add_argument('-r', '--rectify-config', default=None,
                        help='path of the rectification config file')
    args = parser.parse_args()
    _main(args)
