#!/usr/bin/env python3
from pathlib import Path
import cv2
import argparse
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _add_gaussian_blur(args):

    input_path = args.input_image_path
    output_path = args.output_image_path
    output_dir = Path(output_path).parent
    kernel_size = args.gaussian_kernel_size
    sigma = args.gaussian_sigma

    assert Path(input_path).exists(), "Input image path does not exist"
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # Load an image
    image = cv2.imread(input_path)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigma, sigma)

    if output_path is None:
        output_path = output_dir / _append_filename(input_path, 'blurred')

    # Save the image
    cv2.imwrite(output_path, blurred_image)

    assert Path(output_path).exists(), "Failed to generate output image!"

    if args.visualize:
        # Display the images if you want
        cv2.imshow("original", image)
        cv2.imshow("blurred", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _main(args):
    _add_gaussian_blur(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add blur to images.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    parser.add_argument('-k', '--gaussian-kernel-size', default=3,
                        type=int, help='Gaussian blur kernel size')
    parser.add_argument('-s', '--gaussian-sigma', default=2.0,
                        type=float, help='Gaussian blur sigma')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    args = parser.parse_args()

    _main(args)
