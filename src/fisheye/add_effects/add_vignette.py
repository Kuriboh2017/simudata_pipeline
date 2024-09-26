#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
import cv2
import numpy as np


def add_vignette(input_path, output_path, strength):
    """
    Adds a vignette effect to an image.

    :param input_path: Path to the input image.
    :param output_path: Path to save the output image.
    :param strength: Strength of the vignette effect. Ranges from 0 to 1.
        Strength between 0.6 and 0.9 is recommended.
        A smaller strength will result in a darker image.
    """
    img = cv2.imread(str(input_path))
    rows, cols = img.shape[:2]
    mask = np.zeros((rows, cols))
    X_result = cv2.getGaussianKernel(cols, int(strength * cols))
    Y_result = cv2.getGaussianKernel(rows, int(strength * rows))
    mask = np.dot(Y_result, X_result.T)
    mask = mask / mask.max()
    for i in range(3):
        img[..., i] = img[..., i] * mask
    cv2.imwrite(str(output_path), img)


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def get_vignette_strength(seed1, seed2):
    # We may want to use the same seed to have a mostly consistent effect between left and right images.
    # So one seed is the same specified by arguments, while the other seed is truly random by time.
    rng1 = np.random.RandomState(seed1)
    vignette_strength = rng1.uniform(0.62, 0.88)
    rng2 = np.random.RandomState(seed2)
    vignette_strength += rng2.uniform(-0.02, 0.02)
    return vignette_strength


def _main(args):
    input_path = args.input_image_path
    output_path = args.output_image_path
    if args.random:
        seed1 = args.random_seed
        seed2 = 0 if '/cube_front/' in str(input_path) else 1
        strength = get_vignette_strength(seed1, seed2)
    else:
        strength = args.strength
    assert 0 <= strength <= 1, "Strength must be between 0 and 1"
    if output_path is None or output_path in ("None", ""):
        output_path = Path(input_path).parent / \
            _append_filename(input_path, 'vignette')
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    assert Path(input_path).exists(), "Input image path does not exist"

    add_vignette(input_path, output_path, strength)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add vignette to images.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('-s', '--strength', default=0.65,
                        type=float, help='Strength of the vignette effect')
    parser.add_argument('--random-seed', default=0,
                        type=int, help='Random seed')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Whether to randomize the strength')
    args = parser.parse_args()

    _main(args)
