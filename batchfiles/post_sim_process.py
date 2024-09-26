#!/usr/bin/env python3
import argparse
import cv2
import logging
import lz4.frame as lz
import numpy as np
import pickle as pkl
import re
import shutil

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)


sim_pinhole_color_map = {'sky': [191, 105, 112], 'wire': [6, 108, 153]}
sim_fisheye_color_map = {'sky': [224, 172, 177], 'wire': [42, 174, 203]}
target_color_map = {'sky': [180, 130, 70],
                    'wire': [247, 96, 126], 'unlabeled': [0, 0, 0]}
target_gray_map = {'sky': 11, 'wire': 18, 'unlabeled': 0}


def _save_lz4(image_data, path):
    arr = np.ascontiguousarray(image_data)
    data = {
        'arr': lz.compress(arr, compression_level=3),
        'shape': image_data.shape,
        'dtype': image_data.dtype
    }
    with open(path, 'wb') as f:
        pkl.dump(data, f)


def _read_pfm(file):
    """ Read a pfm file """
    with open(file, 'rb') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None
        header = file.readline().rstrip()
        header = str(bytes.decode(header, encoding='utf-8'))
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            assert False, 'Not a PFM file.'
        temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
        if dim_match := re.match(r'^(\d+)\s(\d+)\s$', temp_str):
            width, height = map(int, dim_match.groups())
        else:
            assert False, 'Malformed PFM header.'
        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(file, f'{endian}f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
    return data, scale


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
        print(f"Directory '{src}' has been copied to '{dest}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except shutil.Error as e:
        print(f"Error: {e}")


def move_directory(src, dest):
    try:
        shutil.move(src, dest)
        print(f"Directory '{src}' has been moved to '{dest}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except shutil.Error as e:
        print(f"Error: {e}")


def exec_with_scale(func, files):
    with Pool(cpu_count()) as p:
        p.map(func, files)


def _process_rgb_image(file):
    _logger.info(f'Processing {file}')
    img = Image.open(str(file))
    img = img.transpose(Image.ROTATE_90)  # counterclockwise
    img.save(str(file.with_suffix('.webp')), lossless=True)
    file.unlink()


def _process_rgb_dir(rgb_dir):
    rgb_files = list(Path(rgb_dir).iterdir())
    exec_with_scale(_process_rgb_image, rgb_files)


def _process_depth_image(file):
    _logger.info(f'Processing {file}')
    depth = _read_pfm(str(file))[0]
    depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _save_lz4(depth, str(file.with_suffix('.lz4')))
    file.unlink()


def _process_depth_dir(depth_dir):
    depth_files = list(Path(depth_dir).iterdir())
    exec_with_scale(_process_depth_image, depth_files)


def _process_segmentation_image(file, color_dir, gray_dir):
    _logger.info(f'Processing {file}')
    img = cv2.imread(str(file))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    color_img = img.copy()
    gray_img = np.zeros_like(img[..., 0], dtype=np.uint8)
    unmatched_mask = np.ones_like(gray_img, dtype=bool)
    for label, color in sim_fisheye_color_map.items():
        mask = (img == color).all(axis=-1)
        color_img[mask] = target_color_map[label]
        gray_img[mask] = target_gray_map[label]
        unmatched_mask &= ~mask
    color_img[unmatched_mask] = target_color_map['unlabeled']
    gray_img[unmatched_mask] = target_gray_map['unlabeled']
    color_img = Image.fromarray(color_img)
    color_img.save(str(color_dir / f'{file.stem}.webp'), lossless=False)
    _save_lz4(gray_img, str(gray_dir / f'{file.stem}.lz4'))
    file.unlink()


def _process_segmentation_dir(seg_dir):
    seg_files = list(Path(seg_dir).iterdir())
    folder = Path(seg_dir)
    gray_dir = folder / 'Graymap'
    color_dir = folder / 'Colormap'
    gray_dir.mkdir(exist_ok=True, parents=True)
    color_dir.mkdir(exist_ok=True, parents=True)
    exec_with_scale(partial(_process_segmentation_image,
                    color_dir=color_dir, gray_dir=gray_dir), seg_files)


def _process_cube_dir(cube_cam_dir):
    for cube_dir in Path(cube_cam_dir).iterdir():
        _logger.info(f'Processing {cube_dir}')
        name = cube_dir.name
        if name == 'CubeScene':
            _process_rgb_dir(cube_dir)
        elif name == 'CubeDepth':
            _process_depth_dir(cube_dir)
        elif name == 'CubeSegmentation':
            _process_segmentation_dir(cube_dir)


def _process_dir(output_dir):
    for dir in Path(output_dir).iterdir():
        if dir.is_dir() and dir.name.lower().startswith('cube'):
            _logger.info(f'Processing {dir}')
            _process_cube_dir(dir)


def _main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or f'{Path(input_dir)}_out'
    copy_directory(input_dir, output_dir)
    _process_dir(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('After running Unreal Engine sim, '
                     'rotate images counterclockwise and '
                     'remap segmentation labels.')
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir',
                        help='Directory of the output images')
    args = parser.parse_args()
    _main(args)
