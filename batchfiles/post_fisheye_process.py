#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import argparse
import cv2
import logging
import shutil
import uuid

import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
        print(f"Directory '{src}' has been copied to '{dest}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except shutil.Error as e:
        print(f"Error: {e}")


def _swap_directory_names(path1: Path, path2: Path):
    if not path1.is_dir() or not path2.is_dir():
        raise ValueError("Both paths should point to directories.")
    tmp_dir = path1.parent / str(uuid.uuid4())
    path1.rename(tmp_dir)
    path2.rename(path1)
    tmp_dir.rename(path2)
    _logger.info(f'Swapped {path1} and {path2}')


def _remap_path(relative_path):
    path_parts = list(relative_path.parts)

    for i, part in enumerate(path_parts):
        # Mapping for CubeScene and CubeSegmentation
        if part in ('CubeScene_3to1', 'CubeScene', 'Image_3to1'):
            path_parts[i] = 'Image'
        elif part in ('CubeSegmentation_3to1', 'CubeSegmentation'):
            path_parts[i] = 'Segmentation'
        
        # Mapping for Graymap and Colormap under CubeSegmentation
        if part in ('Graymap_3to1') and path_parts[max(i-1, 0)] == 'Segmentation':
            path_parts[i] = 'Graymap'
        elif part in ('Colormap_3to1') and path_parts[max(i-1, 0)] == 'Segmentation':
            path_parts[i] = 'Colormap'
        
        # Mapping for CubeDepth
        if part in ('CubeDepth_3to1', 'CubeDepth'):
            path_parts[i] = 'Depth'

        if part in ('Disp_3to1', 'Disp'):
            path_parts[i] = 'Disparity'

    return Path(*path_parts)

def _copy_file(file, src_dir, dest_dir):
    relative_path = file.relative_to(src_dir)
    new_relative_path = _remap_path(relative_path)
    new_path = dest_dir / new_relative_path
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(file), str(new_path))
    print(f'Copied {file} to {new_path}')


def _copy_files_if_with_substr(src_dir, dest_dir, substr):
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = src_dir.rglob('*')
    filtered_files = [file for file in files if file.is_file()
                      and substr in file.stem]
    with Pool(cpu_count()) as p:
        p.map(partial(_copy_file, src_dir=src_dir,
              dest_dir=dest_dir), filtered_files)


def _rename_file(file, substr):
    new_name = file.stem.replace(substr, '') + file.suffix
    file.rename(file.parent / new_name)
    print(f'Renamed {file} to {file.parent / new_name}')


def _rename_files_recursive(directory: Path, substr: str):
    files = directory.rglob(f'*{substr}*')
    filtered_files = [file for file in files if file.is_file()
                      and substr in file.stem]
    with Pool(cpu_count()) as p:
        p.map(partial(_rename_file, substr=substr), filtered_files)


def _rename_fisheye_dir(fisheye_dir, output_dir):
    # Rename bottom fisheye to 'group0'
    # Rename up fisheye to 'group1'
    fisheye_dir = fisheye_dir.rename(Path(output_dir) / 'fisheye_all')
    down_fisheye_dir = Path(output_dir) / 'group0'
    up_fisheye_dir = Path(output_dir) / 'group1'
    down_fisheye_dir.mkdir(parents=True, exist_ok=False)
    up_fisheye_dir.mkdir(parents=True, exist_ok=False)
    _copy_files_if_with_substr(fisheye_dir, down_fisheye_dir, '_down')
    _copy_files_if_with_substr(fisheye_dir, up_fisheye_dir, '_up')
    _rename_files_recursive(down_fisheye_dir, '_down')
    _rename_files_recursive(up_fisheye_dir, '_up')
    down_left_dir = down_fisheye_dir / 'cam1_0'
    down_left_dir.rename(down_fisheye_dir / 'cam0_0')
    down_right_dir = down_fisheye_dir / 'cam1_1'
    down_right_dir.rename(down_fisheye_dir / 'cam0_1')
    up_left_dir = up_fisheye_dir / 'cam1_1'
    up_right_dir = up_fisheye_dir / 'cam1_0'
    _swap_directory_names(up_left_dir, up_right_dir)
    shutil.rmtree(fisheye_dir)


def _rename_dirs(output_dir):
    # Rename front pinhole to 'group2'
    pinhole_dir = Path(output_dir) / 'group0'
    if pinhole_dir.exists():
        pinhole_dir = pinhole_dir.rename(Path(output_dir) / 'group2')

    fisheye_dir = Path(output_dir) / 'group1'
    if fisheye_dir.exists():
        _rename_fisheye_dir(fisheye_dir, output_dir)


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


def _rotate_rgb_files(files):
    with Pool(cpu_count()) as p:
        p.map(_rotate_rgb, files)


def _rotate_npz_files(files):
    with Pool(cpu_count()) as p:
        p.map(_rotate_npz, files)


def _process_up_fisheye_dir(up_fisheye_dir):
    assert up_fisheye_dir.exists(
    ), f'Input directory does not exist: {up_fisheye_dir}'
    # Rotate `up-fisheye` images 180 degrees
    files = up_fisheye_dir.rglob('*')
    img_files = [file for file in files if file.is_file()]
    rgb_files = [file for file in img_files if file.name.endswith('.webp')]
    npz_files = [file for file in img_files if file.name.endswith('.npz')]
    if rgb_files:
        _rotate_rgb_files(rgb_files)
    if npz_files:
        _rotate_npz_files(npz_files)

    # Swap left and right images
    cam0_dir = up_fisheye_dir / 'cam1_0'
    cam1_dir = up_fisheye_dir / 'cam1_1'
    if cam0_dir.exists() and cam1_dir.exists():
        _swap_directory_names(cam0_dir, cam1_dir)


def _main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or f'{Path(input_dir)}_renamed'
    copy_directory(input_dir, output_dir)
    _logger.info('Renaming the folders to match the real-world drone data folder structure.')
    _rename_dirs(output_dir)
    _logger.info('Done renaming!')

    _logger.info('Rotating the up fisheye images 180 degrees to let left images has depth and segmentation data.')
    up_fisheye_dir = Path(output_dir) / 'group1'
    _process_up_fisheye_dir(up_fisheye_dir)
    _logger.info('Done rotation!')

    if args.delete_input_dir:
        shutil.rmtree(input_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('After running fisheye conversion, '
                     '(1) rename the folders to match the real-world '
                     'drone data folder structure.'
                     '(2) Rotate the up fisheye images 180 degrees to '
                     'let left images has depth and segmentation data.')
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir',
                        help='Directory of the output images')
    parser.add_argument('-d', '--delete-input-dir', action='store_true',
                        help='Delete the input directory after processing')
    args = parser.parse_args()
    _main(args)
