#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import numpy as np
import os
import pickle as pkl


def crop_erp(data, is_disp, is_seg_gray):
    crop_fov = (185, 120)  # 训练dataloader输出的最终fov
    crop_shape = (1408, 1280)
    # resize
    resize_h = round(crop_shape[0] / crop_fov[0] * 360)
    resize_w = round(crop_shape[1] / crop_fov[1] * 180)
    dsize = (resize_w, resize_h)

    if is_disp:
        resize_scale = resize_w / data.shape[1]
        data = (cv2.resize(data.astype(np.float32), dsize=dsize, interpolation=cv2.INTER_NEAREST)
                * resize_scale).astype(np.float16)  # (resize_h, resize_w, )
    elif is_seg_gray:
        data = cv2.resize(data, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    else:
        data = cv2.resize(data, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    crop_w = crop_shape[1]
    rescale_w = data.shape[1]
    dy = int(round((rescale_w-crop_w)/2))+1
    data = data[:, dy:dy+crop_w]  # (2740, 1920, 3) -> (2740, 1280, 3)

    up_data = np.vstack([data[-19:, :], data[:(1370+19), :]])
    down_data = np.vstack([data[(1370-19):, :], data[:19, :]])
    return down_data, up_data


def append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


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
    assert os.path.exists(
        input_image_path
    ), f"Input image path does not exist: {input_image_path}"
    output_image_path0 = args.output_image_path0
    output_image_path1 = args.output_image_path1
    if output_image_path0 is None or output_image_path0 == 'None':
        output_image_path0 = append_filename(input_image_path, 'erp_0')
    if output_image_path1 is None or output_image_path1 == 'None':
        output_image_path1 = append_filename(input_image_path, 'erp_1')

    if input_image_path.endswith('.npz'):
        data = np.load(input_image_path, allow_pickle=True)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    elif input_image_path.endswith('.lz4'):
        input_img = _read_lz4(input_image_path)
    else:
        input_img = cv2.imread(input_image_path)

    down_img, up_img = crop_erp(
        input_img, is_disp=args.disparity, is_seg_gray=args.segmentation_graymap)
    # Rotate 180 degree to be consistent with the fisheye image conversion
    up_img = np.flip(np.flip(up_img, axis=0), axis=1)

    if output_image_path0.endswith('.lz4'):
        _save_lz4(down_img, output_image_path0)
        _save_lz4(up_img, output_image_path1)
    elif output_image_path0.endswith('.npz'):
        np.savez_compressed(output_image_path0, arr_0=down_img)
        np.savez_compressed(output_image_path1, arr_0=up_img)
    else:
        cv2.imwrite(output_image_path0, np.uint8(down_img))
        cv2.imwrite(output_image_path1, np.uint8(up_img))
    print(f'output_image_path0 = {Path(output_image_path0).absolute()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rectify 1 fisheye image to a erp image.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o0', '--output-image-path0',
                        help='path 0 of the output image')
    parser.add_argument('-o1', '--output-image-path1',
                        help='path 1 of the output image')
    parser.add_argument('-d', '--disparity', action='store_true',
                        help='whether or not processing disparity image')
    parser.add_argument('-g', '--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation grapmap image')
    args = parser.parse_args()
    _main(args)
