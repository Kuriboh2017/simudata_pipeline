#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import math
import numpy as np
import os
import pickle as pkl
from functools import partial
from multiprocessing import Pool, cpu_count

kCurrentDir = os.path.dirname(os.path.abspath(__file__))

RANDOM_RANGE = 10000
TABLE_PREFIX = 'panorama2fisheye'


def _get_rotation(rpy_degrees):
    roll = math.radians(rpy_degrees[0])
    pitch = math.radians(rpy_degrees[1])
    yaw = math.radians(rpy_degrees[2])
    row1 = [math.cos(yaw) * math.cos(pitch),
            math.cos(yaw) * math.sin(pitch) * math.sin(roll) -
            math.sin(yaw) * math.cos(roll),
            math.cos(yaw) * math.sin(pitch) * math.cos(roll) +
            math.sin(yaw) * math.sin(roll)]
    row2 = [math.sin(yaw) * math.cos(pitch),
            math.sin(yaw) * math.sin(pitch) * math.sin(roll) +
            math.cos(yaw) * math.cos(roll),
            math.sin(yaw) * math.sin(pitch) * math.cos(roll) -
            math.cos(yaw) * math.sin(roll)]
    row3 = [-math.sin(pitch), math.cos(pitch) * math.sin(roll),
            math.cos(pitch) * math.cos(roll)]
    return np.array([row1, row2, row3])


def _append_filename(filename, appendix='fisheye'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def _convert_xyz_to_panorama_uv(xyz):
    u = 0.5 / math.pi * math.atan2(xyz[0], -xyz[1]) + 0.5
    v = math.atan2(math.sqrt(xyz[0] * xyz[0] +
                   xyz[1] * xyz[1]), xyz[2]) / math.pi
    return np.array([u, v])


def _unproject_eucm(eucm_alpha, eucm_beta, x, y):
    r2 = x * x + y * y
    beta_r2 = eucm_beta * r2
    term_inside_sqrt = 1 - (2 * eucm_alpha - 1) * beta_r2
    if term_inside_sqrt < 0:
        return np.nan
    numerator = 1 - eucm_alpha * eucm_alpha * beta_r2
    denominator = eucm_alpha * math.sqrt(term_inside_sqrt) + 1 - eucm_alpha
    return numerator / denominator


def generate_p2f_remap_table(width, height, calib_param_src, fxfy_noise, rpy_noise, src_resolution, down):
    width = np.array(width).astype(np.int32)
    height = np.array(height).astype(np.int32)
    calib_param = calib_param_src.copy()

    fx, fy, cx, cy = calib_param[:4]
    fx += fxfy_noise
    fy += fxfy_noise
    eucm_alpha, eucm_beta = calib_param[4], calib_param[5]
    # 90 degree rotation is needed such that the camera is facing the ground (down) or the sky (up)
    pitch = -90 if down else 90
    rmat_up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                                [0, 1, 0],
                                [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])

    rpy_noise_rot = _get_rotation(rpy_noise.copy())
    mapx = np.zeros((height, width), dtype=np.float32)
    mapy = np.zeros((height, width), dtype=np.float32)
    for v in np.arange(height):
        for u in np.arange(width):
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            zn = _unproject_eucm(eucm_alpha, eucm_beta, xn, yn)
            if math.isnan(zn):
                mapx[v, u] = np.nan
                mapy[v, u] = np.nan
                continue
            normal = np.array([xn, yn, zn])
            # It is required to rotate |rpy_noise_rot| before |rmat_up_or_down|.
            normal = np.dot(rpy_noise_rot.T, normal)
            normal = np.dot(rmat_up_or_down.T, normal)
            uv = _convert_xyz_to_panorama_uv(normal)
            mapx[v, u] = uv[0] * src_resolution[0]-0.5
            mapy[v, u] = uv[1] * src_resolution[1]
    return mapy, mapx


def _read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    calibCols = fs.getNode("calibCols").real()
    calibRows = fs.getNode("calibRows").real()
    left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
    right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6

    return calibCols, calibRows, left_intri, right_intri


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


def _generate_table(i, output_dir_str, calibCols, calibRows, intrinsics, src_resolution):
    output_path = Path(output_dir_str) / f'{TABLE_PREFIX}_noise_seed_{i}.npz'
    # 326 is roughly the current fx, fy value, which roughly represents 1 pixel noise
    # Note: the 1 pixel noise is quite small, which is barely visible.
    noise_candidate = i / RANDOM_RANGE / 326
    fxfy_noise = noise_candidate if i % 4 == 1 else 0
    # i / RANDOM_RANGE: 0 ~ 1
    # candidate: -0.1 ~ 0.1
    candidate = (2 * i / RANDOM_RANGE - 1) / 10
    if i % 8 == 1:
        rpy_noise = np.array([candidate, 0.0, 0.0])
    elif i % 8 == 3:
        rpy_noise = np.array([0.0, candidate, 0.0])
    elif i % 8 == 5:
        rpy_noise = np.array([0.0, 0.0, candidate])
    else:
        rpy_noise = np.array([candidate, candidate, candidate])
    down_x, down_y = generate_p2f_remap_table(
        calibCols, calibRows, intrinsics, fxfy_noise, rpy_noise, src_resolution, True)
    up_x, up_y = generate_p2f_remap_table(
        calibCols, calibRows, intrinsics, fxfy_noise, rpy_noise, src_resolution, False)
    np.savez_compressed(output_path, down_x=down_x,
                        down_y=down_y, up_x=up_x, up_y=up_y)
    print(
        f'Generated remapping table to {output_path} with fxfy_noise={fxfy_noise}, rpy_noise={rpy_noise}, src_resolution={src_resolution}')


def _main(args):
    rectify_config = args.rectify_config
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"

    output_folder = Path(args.output_dir)

    # 1x: 2560, 1280
    # 2x: 5120, 2560
    src_resolutions = {
        '1x': [2560, 1280],
        '2x': [5120, 2560],
    }

    calibCols, calibRows, left_intri, right_intri = _read_yaml(rectify_config)

    print('Generating remapping table')
    intrinsics = left_intri
    for res, src_resolution in src_resolutions.items():
        output_dir = output_folder / res
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{TABLE_PREFIX}_noiseless.npz'
        fxfy_noise = 0
        rpy_noise = np.array([0.0, 0.0, 0.0])
        down_x, down_y = generate_p2f_remap_table(
            calibCols, calibRows, intrinsics, fxfy_noise, rpy_noise, src_resolution, True)
        up_x, up_y = generate_p2f_remap_table(
            calibCols, calibRows, intrinsics, fxfy_noise, rpy_noise, src_resolution, False)
        np.savez_compressed(output_path, down_x=down_x,
                            down_y=down_y, up_x=up_x, up_y=up_y)
        print(
            f'Generated remapping table to {output_path} with fxfy_noise={fxfy_noise}, rpy_noise={rpy_noise}, src_resolution={src_resolution}')

        arr = list(range(1, RANDOM_RANGE, 2))
        output_dir_s = str(output_dir)
        with Pool(cpu_count()) as p:
            p.map(partial(_generate_table, output_dir_str=output_dir_s,
                          calibCols=calibCols, calibRows=calibRows, intrinsics=intrinsics,
                          src_resolution=src_resolution),
                  arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate remapping tables from panorama to fisheye.')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('-o', '--output-dir',
                        help='path of the output directory')

    args = parser.parse_args()
    _main(args)
