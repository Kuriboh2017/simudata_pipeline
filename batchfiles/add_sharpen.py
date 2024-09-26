#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
import cv2
import numpy as np


def get_blend_mask(shape, strength=0.7):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    X_result = cv2.getGaussianKernel(cols, int(strength * cols))
    Y_result = cv2.getGaussianKernel(rows, int(strength * rows))
    mask = np.dot(Y_result, X_result.T)
    return mask / mask.max()


def add_sharpen(input_path, output_path, sharpen_strength):
    image = cv2.imread(input_path)
    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 2.)
    laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    blend_mask = get_blend_mask(image.shape[:2])
    for i in range(3):
        laplacian_abs[..., i] = laplacian_abs[..., i] * blend_mask
    blend_sharpened = cv2.addWeighted(
        image, 1.0, laplacian_abs, sharpen_strength, 0)
    cv2.imwrite(str(output_path), blend_sharpened)


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def get_sharpen_strength(seed1, seed2):
    # We may want to use the same seed to have a mostly consistent effect between left and right images.
    # So one seed is the same specified by arguments, while the other seed is truly random by time.
    rng1 = np.random.RandomState(seed1)
    sharpen_strength = rng1.uniform(0.32, 0.58)
    rng2 = np.random.RandomState(seed2)
    sharpen_strength += rng2.uniform(-0.02, 0.02)
    return sharpen_strength


def _main(args):
    input_path = args.input_image_path
    output_path = args.output_image_path
    if args.random:
        seed1 = args.random_seed
        seed2 = 0 if '/cube_front/' in str(input_path) else 1
        sharpen_strength = get_sharpen_strength(seed1, seed2)
    else:
        sharpen_strength = args.sharpen_strength
    assert 0 <= sharpen_strength <= 2, "Sharpen Strength must be between 0 and 2"
    if output_path is None or output_path in ("None", ""):
        output_path = Path(input_path).parent / \
            _append_filename(input_path, 'sharpened')
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    assert Path(input_path).exists(), "Input image path does not exist"

    add_sharpen(input_path, output_path, sharpen_strength)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add sharpening to images.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('-s', '--sharpen-strength', default=0.35,
                        type=float, help='Strength of the sharpen effect')
    parser.add_argument('--random-seed', default=0,
                        type=int, help='Random seed')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Whether to randomize the strength')
    args = parser.parse_args()

    _main(args)
