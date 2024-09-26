from subprocess import run
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
import distutils.spawn
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import re

rgb_left_file_mapping = np.load(
    'train_rgb_left_file_mapping.npz', allow_pickle=True)['rgb_left_file_mapping']
seg_gray_mapping = np.load('train_seg_gray_mapping.npz', allow_pickle=True)[
    'seg_gray_mapping']


kRootPath = '/mnt/115-rbd01/fisheye_dataset/train/data_with_rpy_noise'


def _get_both_paths(origin_path):
    p = Path(origin_path)
    folder = p.parent
    filename = p.stem
    suffix = p.suffix
    p0 = folder / f'{filename}_up{suffix}'
    p1 = folder / f'{filename}_down{suffix}'
    return [p0, p1]


def _get_scene_fisheye_folder_path(origin_path):
    matches = re.findall(
        r'\b[\w_]+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\b', origin_path)
    assert len(
        matches) == 1, f'ERROR unexpected length len(matches) = {len(matches)}'
    scene_folder_name = matches[0]
    return Path(kRootPath) / scene_folder_name / 'group1'


def _get_seg_gray_left_dst_path(origin_path):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    left_path = scene_fisheye_path / 'cam1_0' / 'Segmentation'
    rgb_left_path = left_path / 'Graymap_3to1' / filename
    return _get_both_paths(rgb_left_path)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def _mapback_seg_gray_left(depth_mappings):
    for old_path, new_path in depth_mappings.items():
        old_path_npz = overwrite_suffix(old_path)
        dst_paths = _get_seg_gray_left_dst_path(old_path_npz)
        if Path(old_path).exists():
            dst_paths_str = [str(p) for p in dst_paths]
            folder_src_to_dst[old_path] = dst_paths_str
        else:
            print(
                f'ERROR seg gray image does not exist: \n{new_path}')


folder_src_to_dst = {}
_mapback_seg_gray_left(seg_gray_mapping.item())
np.savez_compressed('folder_seg_gray_src_to_dst.npz',
                    folder_src_to_dst=folder_src_to_dst)
