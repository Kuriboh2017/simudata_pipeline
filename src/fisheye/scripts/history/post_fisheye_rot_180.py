#!/usr/bin/env python3
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import uuid
import cv2
import logging
import numpy as np
import os

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def _rotate_rgb(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.rotate(img, cv2.ROTATE_180)
    rot_img_path = img_path.parent / f'rot_180_{img_path.name}'
    cv2.imwrite(str(rot_img_path), img)
    os.replace(rot_img_path, img_path)
    _logger.info(f'Rotated {img_path}')


def _rotate_npz(img_path):
    depth = np.load(img_path, allow_pickle=True)['arr_0']
    depth = cv2.rotate(depth, cv2.ROTATE_180)
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


def _swap_directory_names(path1: Path, path2: Path):
    if not path1.is_dir() or not path2.is_dir():
        raise ValueError("Both paths should point to directories.")
    tmp_dir = path1.parent / str(uuid.uuid4())
    path1.rename(tmp_dir)
    path2.rename(path1)
    tmp_dir.rename(path2)
    _logger.info(f'Swapped {path1} and {path2}')


def _main(args):
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f'Input directory does not exist: {input_dir}'
    files = input_dir.rglob('*')
    img_files = [file for file in files if file.is_file()]
    rgb_files = [file for file in img_files if file.name.endswith('.webp')]
    npz_files = [file for file in img_files if file.name.endswith('.npz')]
    _rotate_rgb_files(rgb_files)
    _rotate_npz_files(npz_files)

    cam0_dir = input_dir / 'cam1_0'
    cam1_dir = input_dir / 'cam1_1'
    if cam0_dir.exists() and cam1_dir.exists():
        _swap_directory_names(cam0_dir, cam1_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('After running fisheye conversion, '
                     'rotate the up fisheye images 180 degrees to '
                     'let left images has depth and segmentation data.')
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    args = parser.parse_args()
    _main(args)
