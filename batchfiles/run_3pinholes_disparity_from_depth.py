#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import cv2
import itertools
import json
import math
import numpy as np
import os

kCurrentDir = os.path.dirname(os.path.abspath(__file__))


def _depth2disp(depth_file, output_dir, recti_params, args):
    depth_data = np.load(str(depth_file))['arr_0']
    output_path = output_dir / depth_file.name
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    baseline = args.baseline
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
    np.savez_compressed(output_path, arr_0=zDatas)


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


def _main(args):
    rectify_config = args.rectify_config
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"
    if not os.path.exists(rectify_config):
        assert False, f'Error: rectification config file {rectify_config} does not exist'
    l, r, trans, rmat, multispecs = read_yaml(rectify_config)

    assert Path(args.input_dir).exists(
    ), f'Error: depth dir {args.input_dir} does not exist'

    depth_dir = Path(args.input_dir)
    depth_files = sorted(depth_dir.glob('*.npz'))
    output_dir = args.output_dir
    if output_dir is None or output_dir == 'None':
        output_dir = depth_dir.parent / 'Disparity'
    else:
        output_dir = Path(args.output_dir)
    assert not output_dir.exists(
    ), f'Error: output dir {output_dir} already exists'
    output_dir.mkdir(parents=True)
    with open(output_dir / 'baseline.json', 'w') as f:
        baseline = {'baseline_in_meters': args.baseline}
        json.dump(baseline, f)
        print(f'baseline: {args.baseline} meters')

    with Pool(cpu_count()) as p:
        p.map(partial(_depth2disp, output_dir=output_dir,
              recti_params=multispecs, args=args), depth_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert 1 panorama image to the 3-pinholes directly.')
    parser.add_argument('-i', '--input-dir', '--depth-dir', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-dir', '--disparity-dir',
                        help='path of the output disparity image')
    parser.add_argument('-b', '--baseline', type=float, default=0.09,
                        help='baseline of the stereo camera')
    parser.add_argument('-r', '--rectify-config', default=None,
                        help='path of the rectification config file')
    args = parser.parse_args()
    _main(args)
