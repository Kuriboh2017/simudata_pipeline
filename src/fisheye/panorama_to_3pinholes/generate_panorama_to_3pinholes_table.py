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

RANDOM_RANGE = 10000
TABLE_PREFIX = 'panorama2pinholes'


def ConvertXYZToPanoramaUV(xyz):
    x, y, z = xyz
    u = 0.5 / math.pi * math.atan2(x, -y) + 0.5
    v = math.atan2(math.sqrt(x * x + y * y), z) / math.pi
    return np.array([u, v])


class _RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


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
    return [row1, row2, row3]


def generate_p2p_remap_table(recti_specs, rpy_noise, src_resolution, down):
    left_remap_x_list = []
    left_remap_y_list = []
    pitch = -90 if down else 90
    up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                           [0, 1, 0],
                           [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])

    rpy_noise_rot = _get_rotation(rpy_noise.copy())
    for rectispec in recti_specs:
        roll = rectispec.axis
        r_rad = math.radians(roll)
        pinhole63 = np.array([[1, 0, 0],
                              [0, math.cos(r_rad), -math.sin(r_rad)],
                              [0, math.sin(r_rad), math.cos(r_rad)]])

        rot = np.dot(pinhole63, np.dot(rpy_noise_rot, up_or_down))

        left_x, left_y = _remap_camera(
            rectispec.param, rot, rectispec.row, rectispec.col, src_resolution)

        left_remap_x_list.append(left_x)
        left_remap_y_list.append(left_y)

    # concatenate along y axis
    left_remap_x = np.concatenate(left_remap_x_list, axis=0)
    left_remap_y = np.concatenate(left_remap_y_list, axis=0)
    return left_remap_x, left_remap_y


def _remap_camera(recti_param, rmat, recti_row, recti_col, src_resolution):
    irot = rmat.T
    mapx = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    mapy = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    for v in range(int(recti_row)):
        for u in range(int(recti_col)):
            xn = (u - recti_param[2]) / recti_param[0]
            yn = (v - recti_param[3]) / recti_param[1]
            p3d = np.array([xn, yn, 1.])

            p3d_new = np.dot(irot, p3d)

            uv = ConvertXYZToPanoramaUV(p3d_new)
            mapx[v, u] = uv[0] * src_resolution[0]-0.5
            mapy[v, u] = uv[1] * src_resolution[1]
    return mapy, mapx


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


def _generate_table(i, output_dir_str, multispecs, src_resolution):
    output_path = Path(output_dir_str) / f'{TABLE_PREFIX}_noise_seed_{i}.npz'
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
    down_x, down_y = generate_p2p_remap_table(
        multispecs, rpy_noise, src_resolution, True)
    up_x, up_y = generate_p2p_remap_table(
        multispecs, rpy_noise, src_resolution, False)
    np.savez_compressed(output_path, down_x=down_x,
                        down_y=down_y, up_x=up_x, up_y=up_y)
    print(
        f'Generated remapping table to {output_path} with rpy_noise={rpy_noise}, src_resolution={src_resolution}')


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
    l, r, trans, rmat, multispecs = read_yaml(rectify_config)

    print(f'translation = {trans}')
    print(f'rotation = {rmat}')

    for spec in multispecs:
        print(spec.row, spec.col, spec.param, spec.axis)

    print('Generating remapping table')
    for res, src_resolution in src_resolutions.items():
        output_dir = output_folder / res
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{TABLE_PREFIX}_noiseless.npz'
        rpy_noise = np.array([0.0, 0.0, 0.0])
        down_x, down_y = generate_p2p_remap_table(
            multispecs, rpy_noise, src_resolution, True)
        up_x, up_y = generate_p2p_remap_table(
            multispecs, rpy_noise, src_resolution, False)
        np.savez_compressed(output_path, down_x=down_x,
                            down_y=down_y, up_x=up_x, up_y=up_y)
        print(
            f'Generated remapping table to {output_path} with rpy_noise={rpy_noise}, src_resolution={src_resolution}')

        arr = list(range(1, RANDOM_RANGE, 2))
        output_dir_s = str(output_dir)
        with Pool(cpu_count()) as p:
            p.map(partial(_generate_table, output_dir_str=output_dir_s,
                          multispecs=multispecs, src_resolution=src_resolution),
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
