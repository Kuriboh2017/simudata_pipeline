#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import cv2
import math
import numpy as np
import os

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_fisheye2pinholes.npz'

RANDOM_RANGE = 10000
TABLE_PREFIX = 'fisheye2pinholes'


class _RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


def generate_f2p_remap_table(intrinsics, fxfy_noise, recti_specs):
    remap_x_list = []
    remap_y_list = []
    for spec in recti_specs:
        rot = math.radians(spec.axis)
        mat = np.array([[1, 0, 0],
                        [0, math.cos(rot), -math.sin(rot)],
                        [0, math.sin(rot), math.cos(rot)]])
        x, y = _remap_camera(
            intrinsics, fxfy_noise, spec.param, mat, spec.row, spec.col)
        remap_x_list.append(x)
        remap_y_list.append(y)
    # concatenate along y axis
    remap_x = np.concatenate(remap_x_list, axis=0)
    remap_y = np.concatenate(remap_y_list, axis=0)
    return remap_x, remap_y


def _remap_camera(calib_param, fxfy_noise, recti_param, rmat, recti_row, recti_col):
    fx, fy, cx, cy = calib_param[:4]
    fx += fxfy_noise
    fy += fxfy_noise
    eucm_alpha, eucm_beta = calib_param[4], calib_param[5]
    irot = rmat.T
    mapx = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    mapy = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    for v in range(int(recti_row)):
        for u in range(int(recti_col)):
            xn = (u - recti_param[2]) / recti_param[0]
            yn = (v - recti_param[3]) / recti_param[1]
            p3d = np.array([xn, yn, 1.])
            p3d_new = np.dot(irot, p3d)

            x = p3d_new[0]
            y = p3d_new[1]
            z = p3d_new[2]

            r2 = x * x + y * y
            rho2 = eucm_beta * r2 + z * z
            rho = np.sqrt(rho2)
            norm = eucm_alpha * rho + (1. - eucm_alpha) * z
            mx, my = x / norm, y / norm

            mapx[v, u] = fx * mx + cx
            mapy[v, u] = fy * my + cy
    return mapx, mapy


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


def _generate_table(i, output_dir_str, intrinsics, multispecs):
    output_path = Path(output_dir_str) / f'{TABLE_PREFIX}_noise_seed_{i}.npz'
    # 326 is roughly the current fx, fy value, which roughly represents 1 pixel noise
    # Note: the 1 pixel noise is quite small, which is barely visible.
    noise_candidate = i / RANDOM_RANGE / 326
    fxfy_noise = noise_candidate if i % 4 == 1 else 0
    map_x, map_y = generate_f2p_remap_table(intrinsics, fxfy_noise, multispecs)
    np.savez_compressed(output_path, map_x=map_x, map_y=map_y)
    print(
        f'Generated remapping table to {output_path} with fxfy_noise={fxfy_noise}')


def _main(args):
    rectify_config = args.rectify_config
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"

    output_folder = Path(args.output_dir)

    left_intrinsics, r, trans, rmat, multispecs = read_yaml(rectify_config)

    print(f'left intrinsics = {left_intrinsics}')
    print(f'right intrinsics = {r}')
    print(f'translation = {trans}')
    print(f'rotation = {rmat}')

    for spec in multispecs:
        print(spec.row, spec.col, spec.param, spec.axis)

    print('Generating remapping table')
    intrinsics = left_intrinsics
    output_dir = output_folder / 'fisheye_to_3pinholes'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{TABLE_PREFIX}_noiseless.npz'
    fxfy_noise = 0
    map_x, map_y = generate_f2p_remap_table(
        intrinsics, fxfy_noise, multispecs)
    np.savez_compressed(output_path, map_x=map_x, map_y=map_y)
    print(
        f'Generated remapping table to {output_path} with fxfy_noise={fxfy_noise}')

    arr = list(range(1, RANDOM_RANGE, 2))
    output_dir_s = str(output_dir)
    with Pool(cpu_count()) as p:
        p.map(partial(_generate_table, output_dir_str=output_dir_s,
                      intrinsics=intrinsics, multispecs=multispecs),
              arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate remapping tables from panorama to pinholes.')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('-o', '--output-dir',
                        help='path of the output directory')

    args = parser.parse_args()
    _main(args)
