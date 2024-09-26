#!/usr/bin/env python3
from panorama_to_3pinholes import remap_min
from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import math
import numpy as np
import os
import pickle as pkl
import time

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_panorama2fisheye.npz'
kRemappingTableCacheFolder = '/mnt/115-data/S22017/sim/remapping_table_cache'


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


def generate_p2f_remap_table(width, height, calib_param, fxfy_scale, rpy_noise, src_resolution, down):
    fx, fy, cx, cy = calib_param[:4]
    fx *= fxfy_scale
    fy *= fxfy_scale
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


def _main(args):
    input_image_path = args.input_image_path
    assert os.path.exists(input_image_path), \
        "Input image path does not exist: {}".format(input_image_path)
    output_image_path0 = args.output_image_path0
    output_image_path1 = args.output_image_path1
    rectify_config = args.rectify_config
    recalculate = args.recalculate
    rpy_noise = np.array([args.roll, args.pitch, args.yaw])

    if output_image_path0 is None or output_image_path0 == 'None':
        output_image_path0 = _append_filename(input_image_path, 'fisheye_0')
    if output_image_path1 is None or output_image_path1 == 'None':
        output_image_path1 = _append_filename(input_image_path, 'fisheye_1')
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"

    if input_image_path.endswith('.npz'):
        data = np.load(input_image_path, allow_pickle=True)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    elif input_image_path.endswith('.lz4'):
        input_img = _read_lz4(input_image_path)
    else:
        input_img = cv2.imread(input_image_path)

    print(f'input_img shape = {input_img.shape}')
    src_resolution = input_img.shape[:2]

    # Load remapping table if exists
    output_folder = Path(os.path.dirname(output_image_path0))
    output_folder.mkdir(parents=True, exist_ok=True)
    cached_remapping_table_path = output_folder / kRemappingTableFile
    remapping_table_path = None
    if args.overwrite_remapping_filepath is not None:
        assert os.path.exists(
            args.overwrite_remapping_filepath
        ), f"Overwrite remapping table filepath does not exist: {args.overwrite_remapping_filepath}"
        remapping_table_path = Path(args.overwrite_remapping_filepath)
    elif not recalculate and cached_remapping_table_path.exists():
        remapping_table_path = cached_remapping_table_path

    subfolder = '_'.join(str(item) for item in src_resolution)
    folder = Path(kRemappingTableCacheFolder) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    baseline_rotation_remapping_table_path = folder / \
        f'panorama2fisheye_roll_{args.roll:.1f}.npz'
    if not recalculate and not remapping_table_path:
        if args.pitch == 0.0 and args.yaw == 0.0:
            if os.path.exists(baseline_rotation_remapping_table_path):
                remapping_table_path = baseline_rotation_remapping_table_path

    if remapping_table_path:
        with np.load(remapping_table_path) as data:
            down_x = data['down_x']
            down_y = data['down_y']
            up_x = data['up_x']
            up_y = data['up_y']
            assert down_x.shape == down_y.shape == up_x.shape == up_y.shape == (1120, 1120), \
                "Remapping table shape mismatch"

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

    if remapping_table_path:
        print(f'Using remapping table {remapping_table_path}')
    else:
        print('Generating remapping table')
        intrinsics = right_intri if args.use_right_intrinsics else left_intri
        down_x, down_y = generate_p2f_remap_table(
            np.array(calibCols).astype(np.int32), np.array(calibRows).astype(np.int32), intrinsics.copy(), np.array(args.fxfy_scale), rpy_noise.copy(), np.array(src_resolution), True)
        up_x, up_y = generate_p2f_remap_table(
            np.array(calibCols).astype(np.int32), np.array(calibRows).astype(np.int32), intrinsics.copy(), np.array(args.fxfy_scale), rpy_noise.copy(), np.array(src_resolution), False)
        # Save/Cache the remapping table
        np.savez_compressed(cached_remapping_table_path, down_x=down_x,
                            down_y=down_y, up_x=up_x, up_y=up_y)
        if args.pitch == 0.0 and args.yaw == 0.0:
            np.savez_compressed(baseline_rotation_remapping_table_path, down_x=down_x,
                                down_y=down_y, up_x=up_x, up_y=up_y)

    if args.depth:
        down_img = remap_min(input_img, down_x, down_y)
        up_img = remap_min(input_img, up_x, up_y)
    else:
        if args.float32:
            input_img = np.float32(input_img)
        else:
            input_img = input_img.astype(np.uint8)
        operation = cv2.INTER_NEAREST if args.segmentation else cv2.INTER_LINEAR
        down_img = cv2.remap(
            input_img, down_x, down_y, operation, borderMode=cv2.BORDER_REPLICATE)
        up_img = cv2.remap(
            input_img, up_x, up_y, operation, borderMode=cv2.BORDER_REPLICATE)

    if args.depth:
        if not args.float32:
            down_img = down_img.astype(np.float16)
            up_img = up_img.astype(np.float16)
        if output_image_path0.endswith('.lz4'):
            _save_lz4(down_img, output_image_path0)
            _save_lz4(up_img, output_image_path1)
        else:
            np.savez_compressed(output_image_path0, arr_0=down_img)
            np.savez_compressed(output_image_path1, arr_0=up_img)
    else:
        if args.float32:
            np.savez_compressed(overwrite_suffix(output_image_path0), down_img)
            np.savez_compressed(overwrite_suffix(output_image_path1), up_img)
        cv2.imwrite(output_image_path0, np.uint8(down_img))
        cv2.imwrite(output_image_path1, np.uint8(up_img))

    print(
        f'down_img shape = {down_img.shape}, path0 = {output_image_path0}')
    print(
        f'up_img shape = {up_img.shape}, path0 = {output_image_path1}')
    if args.visualize:
        cv2.imshow("Panorama", input_img)
        cv2.imshow("Fisheye down", down_img)
        cv2.imshow("Fisheye up", up_img)
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
    parser.add_argument('--float32', action='store_true',
                        help='whether or not to output floating point images')
    args = parser.parse_args()
    _main(args)
