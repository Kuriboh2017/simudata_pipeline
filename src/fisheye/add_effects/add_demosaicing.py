#!/usr/bin/env python3
from pathlib import Path
import argparse
import colour_demosaicing
import cv2
import numpy as np


def BGR2BGGR(color_image):
    bayer_image = np.zeros_like(color_image[:, :, 0])
    bayer_image[0::2, 0::2] = color_image[0::2, 0::2, 0]
    bayer_image[0::2, 1::2] = color_image[0::2, 1::2, 1]
    bayer_image[1::2, 0::2] = color_image[1::2, 0::2, 1]
    bayer_image[1::2, 1::2] = color_image[1::2, 1::2, 2]
    return bayer_image


def add_demosaicing_effects(img_path, output_path):
    color_image = cv2.imread(img_path)
    bayer_image = BGR2BGGR(color_image)
    demosaiced_image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(
        bayer_image, 'RGGB')
    cv2.imwrite(str(output_path), demosaiced_image)


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _main(args):
    input_path = args.input_image_path
    output_path = args.output_image_path
    if output_path is None or output_path in ("None", ""):
        output_path = Path(input_path).parent / \
            _append_filename(input_path, 'demosaiced')
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    assert Path(input_path).exists(), "Input image path does not exist"
    add_demosaicing_effects(input_path, output_path)

    if args.visualize:
        # Display the images if you want
        cv2.imshow("original", image)
        cv2.imshow("demosaiced", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add demosaicing to images.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the demosaiced image')
    args = parser.parse_args()
    _main(args)
