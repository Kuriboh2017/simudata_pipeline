#!/usr/bin/env python3
from pathlib import Path
from scipy import interpolate
import argparse
import cv2
import itertools
import math
import numpy as np
import os

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_pinholes2panorama.npz'

kPanoramaWidth = 1280
kPanoramaHeight = 2560


def append_filename(filename, appendix='remapped_directly'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def ConvertXYZToPanoramaUV(xyz):
    u = 0.5 / math.pi * math.atan2(xyz[0], -xyz[1]) + 0.5
    v = math.atan2(math.sqrt(xyz[0] * xyz[0] +
                   xyz[1] * xyz[1]), xyz[2]) / math.pi
    return np.array([u, v])


def depth2disp(depth_data, recti_params):
    ori_input_h = 1362
    ori_input_w = 1280
    avg_input_h = 454
    baseline = 0.09
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


def generate_remap_table(recti_specs, rpy_noise, down):
    pitch = 90 if down else -90
    up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                           [0, 1, 0],
                           [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])

    rmat_rpy_noise = np.array(_get_rotation(rpy_noise.copy()))
    assert len(recti_specs) > 0, 'recti_specs is empty'
    recti_spec = recti_specs[0]
    roll = recti_spec.axis
    r_rad = math.radians(roll)
    pinhole63 = np.array([[1, 0, 0],
                          [0, math.cos(r_rad), -math.sin(r_rad)],
                          [0, math.sin(r_rad), math.cos(r_rad)]])

    return _remap_camera(recti_spec, pinhole63, rmat_rpy_noise, up_or_down)


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


def _convert_segmentation_to_graymap(segmentation):
    is_equal = np.all(np.equal(segmentation[:, :, 0], segmentation[:, :, 1]) & np.equal(
        segmentation[:, :, 0], segmentation[:, :, 2]))
    assert is_equal, 'segmentation source image is not graymap'
    seg_graymap = segmentation[:, :, 0]
    seg_graymap[seg_graymap == 1] = 0
    seg_graymap[seg_graymap == 2] = 0
    seg_graymap[seg_graymap == 0] = 1
    seg_graymap[seg_graymap == 11] = 0
    seg_graymap[seg_graymap == 18] = 2
    return seg_graymap


def _convert_panorama_uv_to_xyz(UV):
    angles = (2 * math.pi * (UV[0] + 0.5), math.pi * UV[1])
    val = math.sin(angles[1])
    return [val * math.sin(angles[0]), -val * math.cos(angles[0]), math.cos(angles[1])]


def _remap_camera(recti_spec, pinhole63, rmat_rpy_noise, rmat_up_or_down):
    fx, fy, cx, cy = recti_spec.param
    mapx = np.zeros((int(kPanoramaHeight), int(
        kPanoramaWidth)), dtype=np.float32)
    mapy = np.zeros((int(kPanoramaHeight), int(
        kPanoramaWidth)), dtype=np.float32)
    for v in range(int(kPanoramaHeight)):
        for u in range(int(kPanoramaWidth)):
            un = u / kPanoramaWidth
            vn = (v + 0.5) / kPanoramaHeight
            normal = _convert_panorama_uv_to_xyz([vn, un])
            normal = np.dot(rmat_up_or_down.T, normal)
            angle = math.degrees(math.atan2(normal[1], normal[2]))
            if -94.5 < angle < -31.5:
                normal = np.dot(pinhole63, normal)
            elif 31.5 < angle < 94.5:
                normal = np.dot(pinhole63.T, normal)
            elif not -31.5 < angle < 31.5:
                mapx[v, u] = np.nan
                mapy[v, u] = np.nan
                continue
            normal = np.dot(rmat_rpy_noise.T, normal)
            xn = normal[0] / normal[2]
            yn = normal[1] / normal[2]
            lx = fx * xn + cx
            ly = fy * yn + cy
            if 31.5 < angle < 94.5:
                ly += recti_spec.row * 2
            elif -31.5 < angle < 31.5:
                ly += recti_spec.row
            mapx[v, u] = lx
            mapy[v, u] = ly
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


def _main(args):
    input_image_path = args.input_image_path
    assert os.path.exists(input_image_path), \
        "Input image path does not exist: {}".format(input_image_path)
    rectify_config = args.rectify_config
    recalculate = args.recalculate
    rpy_noise = np.array([args.roll, args.pitch, args.yaw])

    output_image_path = args.output_image_path
    if output_image_path is None or output_image_path == 'None':
        output_image_path = append_filename(input_image_path, 'panorama')

    if args.depth or args.segmentation_graymap:
        output_image_path = overwrite_suffix(output_image_path)
        print(f'Output image path0: {output_image_path}')

    # Load remapping table if exists
    output_folder = Path(os.path.dirname(output_image_path))
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

    if remapping_table_path:
        with np.load(remapping_table_path) as data:
            down_x = data['down_x']
            down_y = data['down_y']
            # print(f'down_x.shape = {down_x.shape}')
            assert down_x.shape == down_y.shape == (2560, 1280), \
                "Remapping table shape mismatch"

    if args.depth:
        data = np.load(input_image_path, allow_pickle=True)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    else:
        if input_image_path.endswith('.npz'):
            data = np.load(input_image_path, allow_pickle=True)
            assert 'arr_0' in data, "Key 'arr_0' not found in the npz file"
            input_img = data['arr_0']
        else:
            input_img = cv2.imread(input_image_path)

    print(f'input_img shape = {input_img.shape}')

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
        print(f'rpy_noise: {rpy_noise}')
        down_x, down_y = generate_remap_table(multispecs, rpy_noise, True)
        # Save/Cache the remapping table
        np.savez_compressed(cached_remapping_table_path,
                            down_x=down_x, down_y=down_y)

    if args.depth:
        down_img = remap_min(input_img, down_x, down_y)
    else:
        if args.float32:
            input_img = np.float32(input_img)
        operation = cv2.INTER_NEAREST if args.segmentation_graymap else cv2.INTER_CUBIC
        down_img = cv2.remap(input_img, down_x, down_y,
                             operation, borderMode=cv2.BORDER_REPLICATE)

    if args.depth:
        down_img = depth2disp(down_img, multispecs)
    elif args.segmentation_graymap:
        down_img = _convert_segmentation_to_graymap(down_img)

    if args.delete_odd_rows:
        if args.depth or args.segmentation_graymap:
            down_img = down_img[::2, :]
        else:
            down_img = down_img[::2, :, :]

    if args.depth:
        np.savez_compressed(output_image_path, arr_0=down_img)
    elif args.segmentation_graymap:
        np.savez_compressed(output_image_path, arr_0=down_img)
    else:
        if args.float32:
            np.savez_compressed(overwrite_suffix(output_image_path), down_img)
        else:
            cv2.imwrite(output_image_path, down_img)

    print(f'down_img shape = {down_img.shape}')
    print(f'output_image_path = {output_image_path}')
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
    parser.add_argument('-o', '--output-image-path',
                        help='path of the output image')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('-g', '--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
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


# # project to 3 cameras, uv0 uv1 uv2
# # u in [0,U-1], v in [0,V-1]
# # c0
# normal0 = np.dot(pinhole63, normal)
# normal0 = np.dot(rmat_rpy_noise.T, normal0)
# xn0 = normal0[0] / normal0[2]
# yn0 = normal0[1] / normal0[2]
# lx0 = fx * xn0 + cx
# ly0 = fy * yn0 + cy
# # c1
# # normal1 = normal
# normal1 = np.dot(rmat_rpy_noise.T, normal)
# xn1 = normal1[0] / normal1[2]
# yn1 = normal1[1] / normal1[2]
# lx1 = fx * xn1 + cx
# ly1 = fy * yn1 + cy
# # c2
# normal2 = np.dot(pinhole63.T, normal)
# normal2 = np.dot(rmat_rpy_noise.T, normal2)
# xn2 = normal2[0] / normal2[2]
# yn2 = normal2[1] / normal2[2]
# lx2 = fx * xn2 + cx
# ly2 = fy * yn2 + cy

# if -0.5 <= lx0 <= (recti_spec.col + 1) and -0.5 <= ly0 <= (recti_spec.row + 1):
#     mapx[v, u] = lx0
#     mapy[v, u] = ly0
# elif -0.5 <= lx1 <= (recti_spec.col + 1) and -0.5 <= ly1 <= (recti_spec.row + 1):
#     mapx[v, u] = lx1
#     mapy[v, u] = ly1 + recti_spec.row
# elif -0.5 <= lx2 <= (recti_spec.col + 1) and -0.5 <= ly2 <= (recti_spec.row + 1):
#     mapx[v, u] = lx2
#     mapy[v, u] = ly2 + recti_spec.row*2
# else:
#     mapx[v, u] = np.nan
#     mapy[v, u] = np.nan
# return mapx, mapy
