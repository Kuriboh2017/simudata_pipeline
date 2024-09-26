#!/usr/bin/env python3
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse


def plot_error(baseline, error_image, name, error_limit, error_scale, width=0):
    origin_error = error_image - baseline
    error = error_image * error_scale - baseline * error_scale
    error[:, :, 2] = np.sqrt(error[:, :, 0]**2 + error[:, :, 1]**2)
    errorc = error.copy()
    error_limit *= error_scale
    errorc[errorc > error_limit] = error_limit
    errorc[errorc < -error_limit] = -error_limit
    for i in range(3):
        plt.clf()
        plt.imshow((errorc[width:, :, i] / error_scale),
                   cmap='RdBu', interpolation='none')
        plt.colorbar()
        error_map = f'{name}_error_{i}'
        plt.savefig(f'{error_map}_color.png')
        error_one = np.abs(error[width:, :, i])
        scaled_error = np.dstack([error_one, error_one, error_one])
        cv2.imwrite(
            f'{error_map}_gray_scale_{int(error_scale)}.png', scaled_error)
        plt.show()
    return origin_error


def _main(args):
    base_image = args.base_image
    error_image = args.error_image
    assert Path(base_image).exists(), f'{base_image} does not exist'
    assert Path(error_image).exists(), f'{error_image} does not exist'
    name = Path(error_image).stem
    if base_image.endswith('.npz'):
        base_image = np.load(base_image)['arr_0']
    else:
        base_image = cv2.imread(base_image)
    if error_image.endswith('.npz'):
        error_image = np.load(error_image)['arr_0']
    else:
        error_image = cv2.imread(error_image)
    error_limit = float(args.error_visual_range)
    error_scale = args.error_visual_scale
    error = plot_error(base_image, error_image, name, error_limit, error_scale)
    np.savez_compressed(f'{name}_error.npz', arr_0=error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='compare two images and plot errors.')
    parser.add_argument('-b', '--base-image', required=True,
                        help='path of the base image')
    parser.add_argument('-e', '--error-image', required=True,
                        help='path of the error image')
    parser.add_argument('-r', '--error-visual-range', type=float, default=5.0,
                        help='visual range of the error')
    parser.add_argument('-s', '--error-visual-scale', type=float, default=10.0,
                        help='error scale')
    args = parser.parse_args()
    _main(args)
