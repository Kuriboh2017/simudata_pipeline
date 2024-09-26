#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import numpy as np
import os
import pickle as pkl
from panorama_to_3pinholes import remap_min

kCurrentDir = os.path.dirname(os.path.abspath(__file__))
kRemappingTableFile = 'lut_fisheye2erp.npz'


def append_filename(filename, appendix='remap_erp'):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_suffix(filename, new_suffix='.npz'):
    path = Path(filename)
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


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
    output_image_path = args.output_image_path
    if output_image_path is None:
        output_image_path = append_filename(input_image_path)

    # Load remapping table if exists
    output_folder = Path(os.path.dirname(output_image_path))
    output_folder.mkdir(parents=True, exist_ok=True)
    remapping_table_path = Path(kCurrentDir) / kRemappingTableFile

    with np.load(remapping_table_path) as data:
        lx, ly, rx, ry = data['lx'], data['ly'], data['rx'], data['ry']

    if input_image_path.endswith('.npz'):
        data = np.load(input_image_path, allow_pickle=True)
        assert 'arr_0' in data, "Key 'arr_0' not found in the depth npz file"
        input_img = data['arr_0']
    elif input_image_path.endswith('.lz4'):
        input_img = _read_lz4(input_image_path)
    else:
        input_img = cv2.imread(input_image_path)

    print(f'input_img shape = {input_img.shape}')
    assert input_img.shape[0] == 1120 and input_img.shape[1] == 1120, \
        'Error: input image dimension is not 1120x1120'

    if args.delete_odd_rows:
        lx = lx[::2]
        ly = ly[::2]

    if args.float32:
        input_img = np.float32(input_img)
    else:
        input_img = input_img.astype(np.uint8)
    operation = cv2.INTER_NEAREST if args.segmentation else cv2.INTER_LINEAR
    output_img = cv2.remap(input_img, lx, ly, operation)

    if args.depth or args.segmentation_graymap:
        if args.depth and not args.float32:
            output_img = output_img.astype(np.float16)
            up_img = up_img.astype(np.float16)
        if output_image_path.endswith('.lz4'):
            _save_lz4(output_img, output_image_path)
        else:
            np.savez_compressed(output_image_path, arr_0=output_img)
    else:
        if args.float32:
            np.savez_compressed(overwrite_suffix(
                output_image_path), output_img)

        cv2.imwrite(output_image_path, output_img.astype(np.uint8))

    print(f'output_img shape = {output_img.shape}')
    print(f'output_image_path = {output_image_path}')
    if args.visualize:
        cv2.imshow("Original", input_img)
        cv2.imshow("Remapped", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rectify 1 fisheye image to a erp image.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-image-path',
                        help='path of the output image')
    parser.add_argument('--output-depth', action='store_true',
                        help='output depth image in addition to disparity')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('--segmentation', action='store_true',
                        help='whether or not processing segmentation image')
    parser.add_argument('-g', '--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('--delete-odd-rows', action='store_true',
                        help='delete odd row of the remapped image')
    parser.add_argument('--overwrite-remapping-filepath',
                        help='overwrite remapping table filepath')
    parser.add_argument('--float32', action='store_true',
                        help='whether or not to output floating point images')
    args = parser.parse_args()
    _main(args)
