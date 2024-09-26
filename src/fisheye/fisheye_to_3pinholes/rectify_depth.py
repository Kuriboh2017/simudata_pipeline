#!/usr/bin/env python3
import argparse
import cv2
import re
import numpy as np
import os
from rectify_rgba import read_yaml, generate_remap_table, append_filename

kCurrentDir = os.path.dirname(os.path.abspath(__file__))


def _read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

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
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        assert False, 'Malformed PFM header.'

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    file.close()

    return data, scale


def _main(args):
    image_img_path = args.input_image_path
    output_image_path = args.output_image_path
    rectify_config = args.rectify_config
    if output_image_path is None:
        output_image_path = append_filename(image_img_path)

    # test image
    input_depth_image, depth_scale = _read_pfm(image_img_path)
    print(f'input_depth_image shape = {input_depth_image.shape}')
    assert input_depth_image.shape[0] == 1120 and input_depth_image.shape[1] == 1120, \
        "Error: input_depth_image shape is not (1120, 1120)"
    assert depth_scale == 1.0, "depth scale is not 1.0"
    if rectify_config is None or rectify_config == 'None':
        rectify_config = f"{kCurrentDir}/rectification.yml"
    if not os.path.exists(rectify_config):
        assert False, f'Error: rectification config file {rectify_config} does not exist'

    l, r, trans, rmat, multispecs = read_yaml(rectify_config)
    print(l, r, trans, rmat)
    for spec in multispecs:
        print(spec.row, spec.col, spec.param, spec.axis)

    lx, ly, rx, ry = generate_remap_table(l, r, trans, rmat, multispecs)

    remapped_depth_image = cv2.remap(
        input_depth_image, lx, ly, cv2.INTER_NEAREST)

    cv2.imwrite(output_image_path, remapped_depth_image)

    print(f'remapped_depth_image shape = {remapped_depth_image.shape}')
    if args.visualize:
        input_depth_normalized = cv2.normalize(
            input_depth_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        remapped_normalized = cv2.normalize(
            remapped_depth_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow("Input-depth-normalized", input_depth_normalized)
        cv2.imshow("Remapped", remapped_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rectify 1 fisheye depth image to 3 pinholes.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-image-path',
                        help='path of the output image')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the rectified image')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    args = parser.parse_args()
    _main(args)
