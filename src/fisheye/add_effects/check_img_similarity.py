#!/usr/bin/env python3
import argparse
from numba import njit
from skimage import io
import cv2
import numpy as np

t_s = 0.8   # Threshold to determine a pixel is similar or not


@njit
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    product /= stds
    return product


def compare(im1, im2, d=1, sample=1, show=False):
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    similar, total = 0, 0
    sh_row, sh_col = im1.shape[0], im1.shape[1]
    correlation = np.zeros((int(sh_row/sample), int(sh_col/sample)))
    total = correlation.shape[0] * correlation.shape[1]
    for x, i in enumerate(range(d, sh_row - (d + 1), sample)):
        for y, j in enumerate(range(d, sh_col - (d + 1), sample)):
            correlation[x, y] = correlation_coefficient(im1[i - d: i + d + 1,
                                                            j - d: j + d + 1],
                                                        im2[i - d: i + d + 1,
                                                            j - d: j + d + 1])
            if correlation[x, y] > t_s:
                similar += 1
    if show:
        io.imshow(correlation, cmap='gray')
        io.show()
    return float(similar/total)


def _main(args):
    left_scene = args.l
    right_scene = args.r
    similarity = compare(left_scene, right_scene)
    print(f'left and right similarity: {similarity}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='check stereo image by warp')
    parser.add_argument('-l', required=True,
                        help='left input image')
    parser.add_argument('-r', required=True,
                        help='right input image')
    args = parser.parse_args()
    _main(args)
