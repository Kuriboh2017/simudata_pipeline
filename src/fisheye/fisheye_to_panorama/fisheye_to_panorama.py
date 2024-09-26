#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import time
import cv2
import math
import numpy as np
from panorama_to_3pinholes import remap_min

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_fisheye2panorama.npz'


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


def _append_filename(filename, appendix='panorama'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _convert_panorama_uv_to_xyz(UV):
    angles = (2 * math.pi * (UV[0] + 0.5), math.pi * UV[1])
    val = math.sin(angles[1])
    return [val * math.sin(angles[0]), -val * math.cos(angles[0]), math.cos(angles[1])]


def _project_eucm(eucm_alpha, eucm_beta, normal):
    x, y, z = normal
    r2 = x * x + y * y
    rho2 = eucm_beta * r2 + z * z
    rho = np.sqrt(rho2)
    norm = eucm_alpha * rho + (1. - eucm_alpha) * z
    mx, my = x / norm, y / norm
    return mx, my


def _generate_remap_table(width, calib_param, rpy_noise, down):
    height = width * 2
    fx, fy, cx, cy = calib_param[:4]
    eucm_alpha, eucm_beta = calib_param[4], calib_param[5]
    # 90 degree rotation is needed such that the camera is facing the ground (down) or the sky (up)
    pitch = 90 if down else -90
    # rpy_noise_rot = _get_rotation(rpy_noise.copy())
    rmat_up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                                [0, 1, 0],
                                [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])
    mapx = np.zeros((int(height), int(width)), dtype=np.float32)
    mapy = np.zeros((int(height), int(width)), dtype=np.float32)
    for v in range(int(height)):
        for u in range(int(width)):
            un = (u + 0.) / width
            vn = (v + .5) / height
            normal = _convert_panorama_uv_to_xyz([vn, un])
            normal = np.dot(rmat_up_or_down.T, normal)
            mx, my = _project_eucm(eucm_alpha, eucm_beta, normal)
            mapx[v, u] = fx * mx + cx
            mapy[v, u] = fy * my + cy
    return mapx, mapy


def _read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    calibCols = fs.getNode("calibCols").real()
    calibRows = fs.getNode("calibRows").real()
    left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
    right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6

    return calibCols, calibRows, left_intri, right_intri


def _main(args):
    input_image_path = args.input_image_path
    assert os.path.exists(input_image_path), \
        "Input image path does not exist: {}".format(input_image_path)
    output_image_path = args.output_image_path
    rectify_config = args.rectify_config
    rpy_noise = np.array([args.roll, args.pitch, args.yaw])

    if output_image_path is None or output_image_path == 'None':
        output_image_path = _append_filename(input_image_path)
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"

    # # Load remapping table if exists
    # output_folder = Path(os.path.dirname(output_image_path))
    # output_folder.mkdir(parents=True, exist_ok=True)
    # remapping_table_path = output_folder / kRemappingTableFile
    # print(f'Remapping table path: {remapping_table_path}')
    # if remapping_table_path.exists():
    #     with np.load(remapping_table_path) as data:
    #         down_x = data['down_x']
    #         down_y = data['down_y']
    #         up_x = data['up_x']
    #         up_y = data['up_y']
    #         assert down_x.shape == down_y.shape == up_x.shape == up_y.shape == (1120, 1120), \
    #             "Remapping table shape mismatch"

    if args.depth:
        data = np.load(input_image_path)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    else:
        input_img = cv2.imread(input_image_path)

    # # Rotate and flip the input image if they are webp or npz from production pipeline.
    if not args.original_sim_image:
        input_img = cv2.flip(input_img, 0)
        #     input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
    print(f'input_img shape = {input_img.shape}')

    calibCols, calibRows, left_intri, right_intri = _read_yaml(rectify_config)
    seed = time.time()
    np.random.seed(int(seed))
    assert len(left_intri) == len(
        right_intri) == 6, 'Error: invalid intrinsics'
    for i in range(6):
        left_intri[i] = np.random.normal(left_intri[i], left_intri[i] / 100.0 *
                                         args.calibration_noise_level)
        right_intri[i] = np.random.normal(right_intri[i], right_intri[i] / 100.0 *
                                          args.calibration_noise_level)

    # if remapping_table_path.exists():
    #     print('Using cached remapping table')
    # else:
    #     print('Generating remapping table')
    intrinsics = right_intri if args.use_right_intrinsics else left_intri
    width = 1280
    down_x, down_y = _generate_remap_table(
        width, intrinsics, rpy_noise.copy(), True)
    # up_x, up_y = _generate_remap_table(
    #     calibCols, calibRows, intrinsics, rpy_noise.copy(), False)

    # # Save/Cache the remapping table
    # np.savez_compressed(remapping_table_path, down_x=down_x,
    #                     down_y=down_y, up_x=up_x, up_y=up_y)

    if args.depth:
        down_img = remap_min(input_img, down_x, down_y)
        # up_img = remap_min(input_img, up_x, up_y)
    else:
        panorama = cv2.remap(
            input_img, down_x, down_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        print(f'panorama shape = {panorama.shape}')
        # up_img = cv2.remap(
        #     input_img, up_x, up_y, operation, borderMode=cv2.BORDER_REPLICATE)

    if not args.original_sim_image:
        panorama = cv2.flip(panorama, 0)
    #     up_img = cv2.flip(up_img, 0)
    if args.depth:
        np.savez_compressed(output_image_path, arr_0=panorama)
    else:
        top_half = panorama[:1280, :]
        bottom_half = panorama[1280:, :]
        panorama = np.concatenate((bottom_half, top_half))
        cv2.imwrite(output_image_path, panorama)

    print(
        f'panorama shape = {panorama.shape}, path0 = {output_image_path}')
    if args.visualize:
        cv2.imshow("fisheye", input_img)
        cv2.imshow("down fisheye remapped panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a fisheye to a panorama.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-image-path',
                        help='path of the output image')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
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
    parser.add_argument('--add-dummy-zeros', action='store_true',
                        help='whether or not adding dummy zeros to the top half of the image')
    args = parser.parse_args()
    _main(args)
