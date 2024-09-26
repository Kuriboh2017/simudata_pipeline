#!/usr/bin/env python3
from lz4.frame import decompress as lzdecompress
from pathlib import Path
import argparse
import cv2
import numpy as np
import pickle as pkl
import re


def _append_filename(filename, suffix=None, appendix='visualize'):
    p = Path(filename)
    suffix = suffix if suffix is not None else p.suffix
    return "{0}_{2}{1}".format(p.stem, suffix, appendix)


def _read_pfm(file):
    """ Read a pfm file """
    with open(file, 'rb') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        header = str(bytes.decode(header, encoding='utf-8'))
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            assert False, 'Not a PFM file.'

        temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
        if dim_match := re.match(r'^(\d+)\s(\d+)\s$', temp_str):
            width, height = map(int, dim_match.groups())
        else:
            assert False, 'Malformed PFM header.'

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, f'{endian}f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
    return data, scale


def _read_lz4(path):
    import lz4.frame as lz
    import pickle as pkl
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr


def load_pkl_disp(disp_path):
    with open(disp_path, "rb") as f_disp_in:
        disp_data = pkl.load(f_disp_in)
        disp = np.frombuffer(lzdecompress(
            disp_data['left_disparity']['data']), dtype=disp_data['left_disparity']['dtype'])
        disp = disp.reshape(disp_data['left_disparity']['shape'])
    return disp


def _main(args):
    if args.input_image_path.endswith('.pfm'):
        data = _read_pfm(args.input_image_path)
        image_data = data[0]
    elif args.input_image_path.endswith('.npz'):
        data = np.load(args.input_image_path)
        image_data = data['arr_0']
    elif args.input_image_path.endswith('.lz4'):
        image_data = _read_lz4(args.input_image_path)
    elif args.input_image_path.endswith('.pkl'):
        image_data = load_pkl_disp(args.input_image_path)

    image_data = image_data.astype(np.float32)
    print(f'image_data.shape: {image_data.shape}')
    # Normalize the depth image
    image_data = cv2.normalize(
        image_data, None, alpha=0.001, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Log transform the depth image
    image_data = np.log(image_data)

    # Normalize the depth image
    image_normalized = cv2.normalize(
        image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply colormap on normalized depth image
    depth_image_colored = cv2.applyColorMap(cv2.convertScaleAbs(
        image_normalized, alpha=255), cv2.COLORMAP_JET)

    cv2.imshow('depth', depth_image_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save_visualization:
        filename = _append_filename(args.input_image_path, '.png')
        print(f'Save visualization image: {filename}')
        cv2.imwrite(filename, depth_image_colored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the depth image.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Path to the input image')
    parser.add_argument('-s', '--save-visualization', action='store_true',
                        help='Save the visualization image')
    args = parser.parse_args()
    _main(args)
