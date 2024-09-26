#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import itertools
import math
import numpy as np
import os
import time
from panorama_to_3pinholes import remap_min

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'remapping_table_fisheye2pinholes.npz'


def append_filename(filename, appendix='remapped'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


class _RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


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
            # Equation is derived with the assumption of 1.0 planner depth.
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


def generate_f2p_remap_table(intrinsics, recti_specs):
    remap_x_list = []
    remap_y_list = []
    for spec in recti_specs:
        rot = math.radians(spec.axis)
        mat = np.array([[1, 0, 0],
                        [0, math.cos(rot), -math.sin(rot)],
                        [0, math.sin(rot), math.cos(rot)]])
        x, y = _remap_camera(
            intrinsics, spec.param, mat, spec.row, spec.col)
        remap_x_list.append(x)
        remap_y_list.append(y)
    # concatenate along y axis
    remap_x = np.concatenate(remap_x_list, axis=0)
    remap_y = np.concatenate(remap_y_list, axis=0)
    return remap_x, remap_y


def _remap_camera(calib_param, recti_param, rmat, recti_row, recti_col):
    fx, fy, cx, cy = calib_param[:4]
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


def _main(args):
    input_image_path = args.input_image_path
    output_image_path = args.output_image_path
    rectify_config = args.rectify_config
    recalculate = args.recalculate
    if output_image_path is None:
        output_image_path = append_filename(input_image_path)

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
            map_x = data['map_x']
            map_y = data['map_y']
            assert map_y.shape == map_x.shape == (1362, 1280), \
                "Remapping table shape mismatch"

    if args.depth:
        data = np.load(input_image_path)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    else:
        if args.float32:
            input_img = np.load(input_image_path, allow_pickle=True)['arr_0']
        else:
            input_img = cv2.imread(input_image_path)

    print(f'input_img shape = {input_img.shape}')
    assert input_img.shape[0] == 1120 and input_img.shape[1] == 1120, \
        'Error: input image dimension is not 1120x1120'
    assert args.calibration_noise_level >= 0.0, \
        'Error: calibration noise level must be non-negative'
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"
    if not os.path.exists(rectify_config):
        assert False, f'Error: rectification config file {rectify_config} does not exist'

    l, r, trans, rmat, multispecs = read_yaml(rectify_config)
    seed = time.time()
    np.random.seed(int(seed))
    assert len(l) == 6 and len(r) == 6, 'Error: invalid intrinsics'
    for i in range(6):
        l[i] = np.random.normal(l[i], l[i] / 100.0 *
                                args.calibration_noise_level)

    print(f'left intrinsics = {l}')
    print(f'right intrinsics = {r}')
    print(f'translation = {trans}')
    print(f'rotation = {rmat}')

    for spec in multispecs:
        print(spec.row, spec.col, spec.param, spec.axis)

    if remapping_table_path:
        print(f'Using remapping table {remapping_table_path}')
    else:
        print('Generating remapping table')
        intrinsics = r if args.use_right_intrinsics else l
        map_x, map_y = generate_f2p_remap_table(intrinsics, multispecs)
        # Save/Cache the remapping table
        np.savez_compressed(cached_remapping_table_path, map_x=map_x,
                            map_y=map_y)

    if args.delete_odd_rows:
        map_x = map_x[::2]
        map_y = map_y[::2]

    if args.depth:
        output_img = remap_min(input_img, map_x, map_y)
        if args.output_depth:
            depth_filename = f'{Path(input_image_path).stem}_depth.npz'
            depth_path = Path(output_image_path).parent / depth_filename
            np.savez_compressed(depth_path, arr_0=output_img)
        output_img = depth2disp(output_img, multispecs)
    else:
        if args.float32:
            input_img = np.float32(input_img)
        else:
            input_img = input_img.astype(np.uint8)
        operation = cv2.INTER_NEAREST if args.segmentation else cv2.INTER_LINEAR
        output_img = cv2.remap(input_img, map_x, map_y, operation)

    if args.depth:
        if not args.float32:
            output_img = output_img.astype(np.float16)
        np.savez_compressed(output_image_path, arr_0=output_img)
    else:
        if args.float32:
            np.savez_compressed(overwrite_suffix(
                output_image_path), output_img)
        cv2.imwrite(output_image_path, output_img)

    print(f'output_img shape = {output_img.shape}')
    if args.visualize:
        cv2.imshow("Original", input_img)
        cv2.imshow("Remapped", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rectify 1 fisheye image to 3 pinholes.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-image-path',
                        help='path of the output image')
    parser.add_argument('--output-depth', action='store_true',
                        help='output depth image in addition to disparity')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('--use-right-intrinsics', action='store_true',
                        help='use right intrinsics instead of left intrinsics')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('--segmentation', action='store_true',
                        help='whether or not processing segmentation image')
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
