#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
from fisheye.utils import customized_remap
from fisheye.core.table_store import TableStore
from fisheye.core.types import (
    CameraType, Rotation, Intrinsics, EucmIntrinsics, ImageParams, RemappingTableParams)

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_panorama2fisheye.npz'


def _append_filename(filename, appendix='fisheye'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, '.png', appendix)


class _RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


def read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    calibCols = fs.getNode("calibCols").real()
    calibRows = fs.getNode("calibRows").real()
    left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
    right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6
    multi_recti_params = []
    multi_recti_params_node = fs.getNode("multiRectis")
    for i in range(multi_recti_params_node.size()):
        n = multi_recti_params_node.at(i)
        tmp = _RectiSpec(n.at(3).real(), n.at(2).real(), [n.at(4).real(), n.at(
            5).real(), n.at(6).real(), n.at(7).real()], n.at(8).real())
        multi_recti_params.append(tmp)
    rotation = fs.getNode("rmat").mat()
    return calibCols, calibRows, left_intri, right_intri, rotation, multi_recti_params


def _main(args):
    input_image_path = args.input_image_path
    assert os.path.exists(input_image_path), \
        "Input image path does not exist: {}".format(input_image_path)
    output_image_path0 = args.output_image_path0
    output_image_path1 = args.output_image_path1
    rectify_config = args.rectify_config
    recalculate = args.recalculate
    rpy_noise = Rotation(args.roll, args.pitch, args.yaw)

    if output_image_path0 is None or output_image_path0 == 'None':
        output_image_path0 = _append_filename(input_image_path, 'fisheye_0')
    if output_image_path1 is None or output_image_path1 == 'None':
        output_image_path1 = _append_filename(input_image_path, 'fisheye_1')
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"

    if args.depth:
        data = np.load(input_image_path, allow_pickle=True)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    else:
        input_img = cv2.imread(input_image_path)

    print(f'input_img shape = {input_img.shape}')
    src_resolution = input_img.shape[:2]

    calibCols, calibRows, left_intri, right_intri, rotation, multi_recti_params = read_yaml(
        rectify_config)
    # intrinsics_left = Intrinsics.from_array(left_intri)
    # intrinsics_right = Intrinsics.from_array(right_intri)
    # seed = time.time()
    # np.random.seed(int(seed))
    # for i in range(6):
    #     left_intri[i] = np.random.normal(left_intri[i], left_intri[i] / 100.0 *
    #                                      args.calibration_noise_level)
    #     right_intri[i] = np.random.normal(right_intri[i], right_intri[i] / 100.0 *
    #                                       args.calibration_noise_level)

    #     src_resolution[0], src_resolution[1], 1120, 1120)

    # intrinsics = intrinsics_right if args.use_right_intrinsics else intrinsics_left

    # src_image_params = ImageParams(
    #     CameraType.PANORAMA, src_resolution[0], src_resolution[1])

    src_intrinsics = EucmIntrinsics.from_array(left_intri)
    src_image_params = ImageParams(CameraType.FISHEYE,
                                   src_resolution[0], src_resolution[1], src_intrinsics)

    # # 3pinholes
    dst_intrinsics = Intrinsics.from_array(multi_recti_params[0].param)
    dst_image_params = ImageParams(
        CameraType.THREE_PINHOLES, 1280, 1362, dst_intrinsics)

    # # fisheye
    # dst_intrinsics = EucmIntrinsics.from_array(left_intri)
    # dst_image_params = ImageParams(
    #     CameraType.FISHEYE, 1120, 1120, dst_intrinsics)

    params = RemappingTableParams(
        src_image_params, dst_image_params, rpy_noise)

    remapping_table = TableStore.get_remapping_table(params)

    down_x, down_y, up_x, up_y = remapping_table
    if args.depth:
        down_img = customized_remap(input_img, down_x, down_y, np.min)
        if up_x is not None and str(up_x).strip() != 'None':
            up_img = customized_remap(input_img, up_x, up_y, np.min)
    else:
        operation = cv2.INTER_NEAREST if args.segmentation else cv2.INTER_LINEAR
        down_img = cv2.remap(
            input_img, down_x, down_y, operation, borderMode=cv2.BORDER_REPLICATE)
        if up_x is not None and str(up_x).strip() != 'None':
            up_img = cv2.remap(
                input_img, up_x, up_y, operation, borderMode=cv2.BORDER_REPLICATE)

    if args.depth:
        np.savez_compressed(output_image_path0, arr_0=down_img)
        if up_x is not None and str(up_x).strip() != 'None':
            np.savez_compressed(output_image_path1, arr_0=up_img)
    else:
        cv2.imwrite(output_image_path0, down_img)
        if up_x is not None and str(up_x).strip() != 'None':
            cv2.imwrite(output_image_path1, up_img)

    print(
        f'down_img shape = {down_img.shape}, path0 = {output_image_path0}')
    if args.visualize:
        cv2.imshow("Panorama", input_img)
        cv2.imshow("Fisheye down", down_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert 1 panorama image to a fisheye.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o0', '--output-image-path0',
                        help='path 0 of the output image')
    parser.add_argument('-o1', '--output-image-path1',
                        help='path 1 of the output image')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('--fxfy-scale', type=float,
                        default=1.0, help='scale fx, fy to simulate lens\' noise')
    parser.add_argument('--use-right-intrinsics', action='store_true',
                        help='use right intrinsics instead of left intrinsics')
    parser.add_argument('--original-sim-image', action='store_true',
                        help='whether or not the input image is from the original sim image')
    parser.add_argument('--roll', type=float, default=0.0,
                        help='extra roll angle in degrees')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='extra pitch angle in degrees')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='extra yaw angle in degrees')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('--segmentation', action='store_true',
                        help='whether or not processing segmentation image')
    parser.add_argument('-l', '--calibration-noise-level', type=float, default=0.0,
                        help='noise level of the camera intrinsic parameters')
    parser.add_argument('--recalculate', action='store_true',
                        help='recalculate remapping table, ignoring cache')
    parser.add_argument('--overwrite-remapping-filepath',
                        help='overwrite remapping table filepath')
    args = parser.parse_args()
    _main(args)
