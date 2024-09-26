#!/usr/bin/env python3
from pathlib import Path
from scipy import interpolate
import argparse
import cv2
import itertools
import lz4.frame as lz
import math
import numpy as np
import os
import pickle as pkl

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_panorama2pinholes.npz'
kRemappingTableCacheFolder = '/mnt/115-data/S22017/sim/remapping_table_cache'


def append_filename(filename, appendix='remapped_directly'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def ConvertXYZToPanoramaUV(xyz):
    x, y, z = xyz
    u = 0.5 / math.pi * math.atan2(x, -y) + 0.5
    v = math.atan2(math.sqrt(x * x + y * y), z) / math.pi
    return np.array([u, v])


def remap_min(input_img, col_data, row_data):
    col_data = _fill_nan_with_border_values(col_data.copy())
    row_data = _fill_nan_with_border_values(row_data.copy())
    col_data_lbound = np.floor(col_data).astype(int)
    col_data_ubound = np.ceil(col_data).astype(int)
    row_data_lbound = np.floor(row_data).astype(int)
    row_data_ubound = np.ceil(row_data).astype(int)

    # Handle out of bounds (e.g. image boundary 1280, 2560) indices
    col_data_lbound = _replace_boundary(col_data_lbound.copy())
    row_data_lbound = _replace_boundary(row_data_lbound.copy())
    col_data_ubound = _replace_boundary(col_data_ubound.copy())
    row_data_ubound = _replace_boundary(row_data_ubound.copy())
    depth_left_top = input_img[row_data_lbound, col_data_lbound]
    depth_left_bot = input_img[row_data_ubound, col_data_lbound]
    depth_right_top = input_img[row_data_lbound, col_data_ubound]
    depth_right_bot = input_img[row_data_ubound, col_data_ubound]
    depth_grid_in_channel = np.stack([
        depth_left_top, depth_left_bot, depth_right_top, depth_right_bot
    ])
    return np.min(depth_grid_in_channel, axis=0)


def depth2disp(depth_data, recti_params, baseline):
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    zDatas = np.zeros([ori_input_h, ori_input_w], dtype=np.float32)
    partZDatas = np.zeros([avg_input_h, ori_input_w], dtype=np.float32)
    for k in range(3):
        k_start = k * avg_input_h
        k_end = k_start + avg_input_h
        partDepth = depth_data[k_start:k_end, :]
        p = recti_params[k].param
        for j, i in itertools.product(range(avg_input_h), range(ori_input_w)):
            # The formula is derived with the assumption of 1.0 planner depth
            # https://blog.csdn.net/long630576366/article/details/125134477
            x = math.pow((i - p[2]), 2) / math.pow(p[0], 2)
            y = math.pow((j - p[3]), 2) / math.pow(p[1], 2)
            # zData here is actually the 1. / cos(theta), where theta is the angle between the
            # center ray and the radial depth ray.
            zData = math.sqrt(math.pow(partDepth[j, i], 2) / (x + y + 1))
            partZDatas[j, i] = zData
        zDatas[k_start:k_end, :] = partZDatas
    zDatas = baseline * p[0] / zDatas
    return zDatas


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


def _fill_nan_with_border_values(arr):
    non_nan_indices = np.argwhere(~np.isnan(arr))
    non_nan_values = arr[~np.isnan(arr)]
    nan_indices = np.argwhere(np.isnan(arr))
    filled_nan_values = interpolate.griddata(
        non_nan_indices, non_nan_values, nan_indices, method='nearest')
    arr[np.isnan(arr)] = filled_nan_values
    return arr


def _replace_boundary(arr):
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)
    arr[max_indices] = max_val - 1
    return arr


def _customized_remap(input_img, col_data, row_data, func):
    col_data = _fill_nan_with_border_values(col_data.copy())
    row_data = _fill_nan_with_border_values(row_data.copy())
    col_data_lbound = np.floor(col_data).astype(int)
    col_data_ubound = np.ceil(col_data).astype(int)
    row_data_lbound = np.floor(row_data).astype(int)
    row_data_ubound = np.ceil(row_data).astype(int)

    # Handle out of bounds (e.g. image boundary 1280, 2560) indices
    col_data_lbound = _replace_boundary(col_data_lbound.copy())
    row_data_lbound = _replace_boundary(row_data_lbound.copy())
    col_data_ubound = _replace_boundary(col_data_ubound.copy())
    row_data_ubound = _replace_boundary(row_data_ubound.copy())
    depth_left_top = input_img[row_data_lbound, col_data_lbound]
    depth_left_bot = input_img[row_data_ubound, col_data_lbound]
    depth_right_top = input_img[row_data_lbound, col_data_ubound]
    depth_right_bot = input_img[row_data_ubound, col_data_ubound]
    depth_grid_in_channel = np.stack([
        depth_left_top, depth_left_bot, depth_right_top, depth_right_bot
    ])
    return func(depth_grid_in_channel, axis=0)


def _remap_segmentation_id(seg_graymap):
    seg_graymap = seg_graymap.copy()
    seg_graymap[seg_graymap == 1] = 0
    seg_graymap[seg_graymap == 2] = 0
    seg_graymap[seg_graymap == 0] = 1
    seg_graymap[seg_graymap == 11] = 0
    seg_graymap[seg_graymap == 18] = 2
    return seg_graymap


def _convert_segmentation_to_graymap(segmentation):
    is_equal = np.all(np.equal(segmentation[:, :, 0], segmentation[:, :, 1]) & np.equal(
        segmentation[:, :, 0], segmentation[:, :, 2]))
    assert is_equal, 'segmentation source image is not graymap'
    return segmentation[:, :, 0]


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
    rectify_config = args.rectify_config
    recalculate = args.recalculate
    rpy_noise = np.array([args.roll, args.pitch, args.yaw])

    output_image_path0 = args.output_image_path0
    output_image_path1 = args.output_image_path1
    if output_image_path0 is None or output_image_path0 == 'None':
        output_image_path0 = append_filename(input_image_path, '3pinholes_0')
    if output_image_path1 is None or output_image_path1 == 'None':
        output_image_path1 = append_filename(input_image_path, '3pinholes_1')

    if args.depth or args.segmentation_graymap:
        suffix = Path(input_image_path).suffix
        output_image_path0 = overwrite_suffix(output_image_path0, suffix)
        output_image_path1 = overwrite_suffix(output_image_path1, suffix)

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
        f'panorama2pinholes_roll_{args.roll:.1f}.npz'
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
            assert down_x.shape == down_y.shape == up_x.shape == up_y.shape == (1362, 1280), \
                "Remapping table shape mismatch"

    assert args.calibration_noise_level >= 0.0, \
        'Error: calibration noise level must be non-negative'
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"
    if not os.path.exists(rectify_config):
        assert False, f'Error: rectification config file {rectify_config} does not exist'

    l, r, trans, rmat, multispecs = read_yaml(rectify_config)

    print(f'translation = {trans}')
    print(f'rotation = {rmat}')

    for spec in multispecs:
        print(spec.row, spec.col, spec.param, spec.axis)

    if remapping_table_path:
        print(f'Using remapping table {remapping_table_path}')
    else:
        print('Generating remapping table')
        down_x, down_y = generate_p2p_remap_table(
            multispecs, rpy_noise, src_resolution, True)
        up_x, up_y = generate_p2p_remap_table(
            multispecs, rpy_noise, src_resolution, False)
        # Save/Cache the remapping table
        np.savez_compressed(cached_remapping_table_path, down_x=down_x,
                            down_y=down_y, up_x=up_x, up_y=up_y)
        if args.pitch == 0.0 and args.yaw == 0.0:
            np.savez_compressed(baseline_rotation_remapping_table_path, down_x=down_x,
                                down_y=down_y, up_x=up_x, up_y=up_y)

    if args.depth:
        down_img = _customized_remap(input_img, down_x, down_y, np.min)
        up_img = _customized_remap(input_img, up_x, up_y, np.min)
    elif args.segmentation_graymap:
        # Remap segmentation id first and use max operation which prefers wires.
        input_img = _remap_segmentation_id(input_img)
        down_img = _customized_remap(input_img, down_x, down_y, np.max)
        up_img = _customized_remap(input_img, up_x, up_y, np.max)
        if down_img.ndim == 3:
            down_img = _convert_segmentation_to_graymap(down_img)
        if up_img.ndim == 3:
            up_img = _convert_segmentation_to_graymap(up_img)
    else:
        if args.float32:
            input_img = np.float32(input_img)
        else:
            input_img = input_img.astype(np.uint8)
        down_img = cv2.remap(input_img, down_x, down_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        up_img = cv2.remap(input_img, up_x, up_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    if args.depth:
        if not args.output_depth:
            down_img = depth2disp(down_img, multispecs, args.down_baseline)
            up_img = depth2disp(up_img, multispecs, args.up_baseline)

    if args.delete_odd_rows:
        if args.depth or args.segmentation_graymap:
            down_img = down_img[::2, :]
            up_img = up_img[::2, :]
        else:
            down_img = down_img[::2, :, :]
            up_img = up_img[::2, :, :]

    if args.depth or args.segmentation_graymap:
        if args.depth and not args.float32:
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

        cv2.imwrite(output_image_path0, down_img.astype(np.uint8))
        cv2.imwrite(output_image_path1, up_img.astype(np.uint8))

    print(f'down_img shape = {down_img.shape}')
    print(f'output_image_path0 = {output_image_path0}')
    if args.visualize:
        cv2.imshow("Original", input_img)
        cv2.imshow("Remapped", down_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert 1 panorama image to the 3-pinholes directly.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o0', '--output-image-path0',
                        help='path 0 of the output image')
    parser.add_argument('-o1', '--output-image-path1',
                        help='path 1 of the output image')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('-g', '--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    parser.add_argument('--output-depth', action='store_true',
                        help='whether output depth or disparity image')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('--original-sim-image', action='store_true',
                        help='whether or not the input image is from the original sim image')
    parser.add_argument('--roll', type=float, default=0.0,
                        help='extra roll angle in degrees')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='extra pitch angle in degrees')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='extra yaw angle in degrees')
    parser.add_argument('-ub', '--up-baseline', type=float, default=0.09,
                        help='up camera baseline')
    parser.add_argument('-db', '--down-baseline', type=float, default=0.105,
                        help='down camera baseline')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('-l', '--calibration-noise-level', type=float, default=0.0,
                        help='noise level of the camera intrinsic parameters')
    parser.add_argument('--recalculate', action='store_true',
                        help='recalculate remapping table, ignoring cache')
    parser.add_argument('--delete-odd-rows', action='store_true',
                        help='delete odd row of the remapped image')
    parser.add_argument('--overwrite-remapping-filepath',
                        help='overwrite remapping table filepath')
    parser.add_argument('--float32', action='store_true',
                        help='whether or not to output floating point images')
    args = parser.parse_args()
    _main(args)
