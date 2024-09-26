#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import numpy as np
import pickle as pkl


def _append_filename(filename, suffix=None, appendix='visualize'):
    p = Path(filename)
    suffix = suffix if suffix is not None else p.suffix
    return "{0}_{2}{1}".format(p.stem, suffix, appendix)


def _read_lz4(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr


def _main(args):
    if args.input_image_path.endswith('.npz'):
        data = np.load(args.input_image_path, allow_pickle=True)
        image_data = data['arr_0']
    elif args.input_image_path.endswith('.lz4'):
        image_data = _read_lz4(args.input_image_path)

    # Apply colormap on normalized depth image
    gray_image_colored = cv2.applyColorMap(cv2.convertScaleAbs(
        image_data, alpha=255), cv2.COLORMAP_JET)

    cv2.imshow('segmentation_gray', gray_image_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save_visualization:
        filename = _append_filename(args.input_image_path, '.png')
        print(f'Save visualization image: {filename}')
        cv2.imwrite(filename, gray_image_colored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the segmentation gray image.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Path to the input image')
    parser.add_argument('-s', '--save-visualization', action='store_true',
                        help='Save the visualization image')
    args = parser.parse_args()
    _main(args)
