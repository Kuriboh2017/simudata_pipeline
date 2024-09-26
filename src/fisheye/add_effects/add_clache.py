#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np


def add_clache(input_path, output_path):
    image = cv2.imread(input_path)
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
    image_clahe = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(str(output_path), image_clahe)


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _main(args):
    input_path = args.input_image_path
    output_path = args.output_image_path
    if output_path is None or output_path in ("None", ""):
        output_path = Path(input_path).parent / \
            _append_filename(input_path, 'clache')
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    assert Path(input_path).exists(), "Input image path does not exist"

    add_clache(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add clache to images.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    args = parser.parse_args()

    _main(args)
