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
depth_file_mapping = np.load('train_depth_file_mapping.npz', allow_pickle=True)[
    'depth_file_mapping']
seg_gray_mapping = np.load('train_seg_gray_mapping.npz', allow_pickle=True)[
    'seg_gray_mapping']
rgb_right_file_mapping = np.load(
    'train_rgb_right_file_mapping.npz', allow_pickle=True)['rgb_right_file_mapping']


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


def _get_rgb_left_dst_path(origin_path):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    left_path = scene_fisheye_path / 'cam1_0'
    rgb_left_path = left_path / 'Image_3to1' / filename
    return _get_both_paths(rgb_left_path)


def _get_seg_color_left_dst_path(origin_path):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    left_path = scene_fisheye_path / 'cam1_0' / 'Segmentation'
    rgb_left_path = left_path / 'Colormap_3to1' / filename
    return _get_both_paths(rgb_left_path)


def _get_seg_gray_left_dst_path(origin_path):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    left_path = scene_fisheye_path / 'cam1_0' / 'Segmentation'
    rgb_left_path = left_path / 'Graymap_3to1' / filename
    return _get_both_paths(rgb_left_path)


def _get_disparity_left_dst_path(origin_path):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    left_path = scene_fisheye_path / 'cam1_0'
    rgb_left_path = left_path / 'Disp_3to1' / filename
    return _get_both_paths(rgb_left_path)


def _get_seg_color_right_dst_path(origin_path, noisy=False):
    scene_fisheye_path = _get_scene_fisheye_folder_path(origin_path)
    filename = Path(origin_path).name
    right_path = scene_fisheye_path / 'cam1_1'
    if noisy:
        rgb_right_path = right_path / 'Image_3to1_noisy' / filename
    else:
        rgb_right_path = right_path / 'Image_3to1' / filename
    return _get_both_paths(rgb_right_path)


def _get_filelist_path(new_path):
    new_paths = _get_both_paths(new_path)
    if not new_paths[0].exists() or not new_paths[1].exists():
        not_exist_files.append(new_paths[0])
        not_exist_files.append(new_paths[1])
        return None
    return new_paths


src_to_dst = {}


def _copy_files(src_paths, dst_paths):
    assert len(src_paths) == len(dst_paths) == 2, 'ERROR unexpected length'
    dst_paths[0].parent.mkdir(parents=True, exist_ok=True)
    # shutil.copy2(src_paths[0], dst_paths[0])
    # shutil.copy2(src_paths[1], dst_paths[1])
    src_to_dst[src_paths[0]] = dst_paths[0]
    src_to_dst[src_paths[1]] = dst_paths[1]


def _mapback_rgb_and_seg_color_left(rgb_left_mappings):
    for old_path, new_path in rgb_left_mappings.items():
        if 'segmentation_color' in str(new_path):
            dst_paths = _get_seg_color_left_dst_path(old_path)
        else:
            dst_paths = _get_rgb_left_dst_path(old_path)
        if src_paths := _get_filelist_path(new_path):
            _copy_files(src_paths, dst_paths)
        else:
            print(
                f'ERROR rgb or seg color left image does not exist: \n{new_path}')


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def _mapback_depth_left(depth_mappings):
    for old_path, new_path in depth_mappings.items():
        dst_paths = _get_disparity_left_dst_path(old_path)
        new_path = new_path.replace('/data/', '/data_depth/')
        if src_paths := _get_filelist_path(new_path):
            _copy_files(src_paths, dst_paths)
        else:
            print(
                f'ERROR depth left image does not exist: \n{new_path}')


def _mapback_seg_gray_left(depth_mappings):
    for old_path, new_path in depth_mappings.items():
        old_path = overwrite_suffix(old_path)
        dst_paths = _get_seg_gray_left_dst_path(old_path)
        new_path = new_path.replace('/data/', '/data_segmentation_gray/')
        if src_paths := _get_filelist_path(new_path):
            _copy_files(src_paths, dst_paths)
        else:
            print(
                f'ERROR seg gray image does not exist: \n{new_path}')


def _mapback_right_rgb(rgb_right_file_mapping):
    for old_path, new_path in rgb_right_file_mapping.items():
        dst_paths = _get_seg_color_right_dst_path(old_path)
        new_path_folder = Path(new_path).parent / 'data'
        new_path = new_path_folder / Path(new_path).name
        if src_paths := _get_filelist_path(new_path):
            _copy_files(src_paths, dst_paths)
        else:
            print(
                f'ERROR right rgb image does not exist: \n{new_path}')


def _find_noisy_files(src_path):
    ids = [1, 2, 4, 7]
    for id in ids:
        path_folder = Path(src_path).parent / f'data_noisy_{id}'
        path = path_folder / Path(src_path).name
        new_paths = _get_filelist_path(path)
        if new_paths and new_paths[0].exists():
            return new_paths
    return None


def _get_rpy(new_path):
    j = Path(new_path).parent / 'rpy_record.json'
    assert j.exists(), 'ERROR rpy_record.json not exist'
    with open(j, 'r') as f:
        rpy = json.load(f)
    return rpy


def _mapback_right_rgb_noisy(rgb_right_file_mapping):
    for old_path, new_path in rgb_right_file_mapping.items():
        dst_paths = _get_seg_color_right_dst_path(old_path, noisy=True)
        src_paths = _find_noisy_files(new_path)
        if not src_paths:
            print('ERROR noisy file not exist: ', new_path)
            continue
        rpy = _get_rpy(src_paths[0])
        _copy_files(src_paths, dst_paths)
        image_to_rpy[str(dst_paths[0])] = rpy
        image_to_rpy[str(dst_paths[1])] = rpy


not_exist_files = []
_mapback_rgb_and_seg_color_left(rgb_left_file_mapping.item())
np.savez_compressed('not_exist_seg_color_left.npz',
                    not_exist_files=not_exist_files)

_mapback_depth_left(depth_file_mapping.item())

# not_exist_files = []
# _mapback_seg_gray_left(seg_gray_mapping.item())
# np.savez_compressed('not_exist_seg_gray_left.npz',
#                     not_exist_files=not_exist_files)

_mapback_right_rgb(rgb_right_file_mapping.item())
image_to_rpy = {}
_mapback_right_rgb_noisy(rgb_right_file_mapping.item())

src_to_dst_str = {str(src): str(dst) for src, dst in src_to_dst.items()}
np.savez_compressed('by_file_src_to_dst_without_seg.npz',
                    src_to_dst=src_to_dst_str)

src_to_dst_arr = np.array(list(src_to_dst_str.items()))


def clean_dict(rpy_dict):
    return {k: {key: value for key, value in v.items() if value is not None}
            for k, v in rpy_dict.items() if v is not None}


image_to_rpy_c = clean_dict(image_to_rpy)
np.savez_compressed('image_path_to_rpy.npz', image_path_to_rpy=image_to_rpy_c)

# img = '/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/data_right/part1/batch83/14646_cam1_1_image.webp'
# paths = _find_noisy_files(img)
# _get_rpy(paths[0])
# old_path = '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-13-47-01/group1/cam1_0/Image/1668656398991.webp'
# new_path = '/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/data/part0/batch0/0_cam1_0_image.webp'

not_exist_seg_color = np.load('not_exist_seg_color_left.npz', allow_pickle=True)[
    'not_exist_files']

not_exist_seg_color = [str(p) for p in not_exist_seg_color]

not_exist_seg_color_filenames = [Path(p).name for p in not_exist_seg_color]

by_file_src_to_dst = np.load('by_file_src_to_dst.npz', allow_pickle=True)[
    'src_to_dst'].item()

file_src_to_dst = {Path(src).name: dst for src,
                   dst in by_file_src_to_dst.items()}

rgb_left_filename_mapping = {
    Path(v).name: k for k, v in rgb_left_file_mapping.item().items()}

not_exist_seg_color_files = [
    src.replace('_down', '')
    for src in not_exist_seg_color_filenames
    if '_down' in src
]
not_exist_origin = [rgb_left_filename_mapping[src]
                    for src in not_exist_seg_color_files if src in rgb_left_filename_mapping]

for f in not_exist_origin:
    if Path(f).exists():
        print(f'old_path exists: {f}')

np.savez_compressed('not_exist_seg_color_origin.npz',
                    not_exist_files=not_exist_origin)

not_exist_seg_color_origin = not_exist_origin


def append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


origin_seg_path = '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-12-18-11/group1/cam1_0/Segmentation/Colormap/1668565856863.webp'


def _get_related_images(origin_seg_path):
    matches = re.findall(
        r'\b[\w_]+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\b', origin_seg_path)
    folder_name = np.unique(matches)[0]
    # left
    cam1_0 = Path(kRootPath) / folder_name / 'group1' / 'cam1_0'
    name = Path(origin_seg_path).name
    name0 = append_filename(origin_seg_path, 'down')
    name1 = append_filename(origin_seg_path, 'up')
    dst_left_color0 = cam1_0 / 'Image_3to1' / name0
    dst_left_color1 = cam1_0 / 'Image_3to1' / name1
    dst_depth0 = cam1_0 / 'Disp_3to1' / name0
    dst_depth1 = cam1_0 / 'Disp_3to1' / name1
    # right
    cam1_1 = Path(cam1_0).parent / 'cam1_1'
    dst_right_color0 = cam1_1 / 'Image_3to1' / name0
    dst_right_color1 = cam1_1 / 'Image_3to1' / name1
    dst_right_noisy_color0 = cam1_1 / 'Image_3to1_noisy' / name0
    dst_right_noisy_color1 = cam1_1 / 'Image_3to1_noisy' / name1
    return [dst_left_color0, dst_left_color1, dst_depth0, dst_depth1,
            dst_right_color0, dst_right_color1, dst_right_noisy_color0, dst_right_noisy_color1]


related_images = [_get_related_images(p) for p in not_exist_origin]
flattened_result = [item for sublist in related_images for item in sublist]
len(flattened_result)

for p in flattened_result:
    if not p.exists():
        print(f'{str(p)} not exists')
    else:
        print(f'removing {str(p)}')
        p.unlink()


origin_seg_path = '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-12-18-11/group1/cam1_0/Segmentation/Colormap/1668565856863.webp'


def _get_related_images(origin_seg_path):
    # left
    cam1_0 = Path(origin_seg_path).parent.parent.parent
    name = Path(origin_seg_path).name
    dst_left_color = cam1_0 / 'Image' / name
    dst_depth = Path(overwrite_suffix(str(cam1_0 / 'Depth' / name)))
    # right
    cam1_1 = Path(cam1_0).parent / 'cam1_1'
    dst_right_color = cam1_1 / 'Image' / name
    return [dst_left_color, dst_depth, dst_right_color]


related_images = [_get_related_images(p) for p in not_exist_origin]
flattened_result = [item for sublist in related_images for item in sublist]
len(flattened_result)
