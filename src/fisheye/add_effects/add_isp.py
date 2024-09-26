#!/usr/bin/env python3
from pathlib import Path
from isp_filters import all_filters
import argparse
import cv2
import numpy as np
import os
import time


def _apply_filters(image, filter_config):
    for filter_dict in filter_config:
        filter_id = filter_dict['filter_id']
        parameters = filter_dict['parameters']
        filter_ = all_filters[filter_id]()
        filter_.parameters = parameters
        image = filter_.apply(image)
    return image


def get_random_mode_config(seed1, seed2):
    # We may want to use the same seed to have a mostly consistent effect between left and right images.
    # So one seed is the same specified by arguments, while the other seed is truly random by time.
    rng1 = np.random.RandomState(seed1)
    randi1 = rng1.randint
    rng2 = np.random.RandomState(seed2)
    randi2 = rng2.randint
    return [{"filter_id": 0, "parameters": [
        randi1(45, 52) + randi2(-1, 1)]},
        {"filter_id": 1, "parameters": [
            randi1(45, 58) + randi2(-1, 1)]},
        {"filter_id": 2, "parameters": [
            randi1(3, 58) + randi2(-1, 1)]},
        {"filter_id": 4, "parameters": [
            randi1(7, 58) + randi2(-1, 1)]},
        {"filter_id": 6, "parameters": [
            randi1(45, 58) + randi2(-1, 1),
            randi1(45, 58) + randi2(-1, 1),
            randi1(45, 58) + randi2(-1, 1)]},
    ]


def get_dark_mode_config(seed1, seed2):
    # We may want to use the same seed to have a mostly consistent effect between left and right images.
    # So one seed is the same specified by arguments, while the other seed is truly random by time.
    rng1 = np.random.RandomState(seed1)
    randi1 = rng1.randint
    rng2 = np.random.RandomState(seed2)
    randi2 = rng2.randint
    return [{"filter_id": 0, "parameters": [
            randi1(42, 46) + randi2(-1, 1)]},
            {"filter_id": 1, "parameters": [
                             randi1(42, 46) + randi2(-1, 1)]},
            {"filter_id": 2, "parameters": [
                             randi1(3, 97) + randi2(-1, 1)]},
            {"filter_id": 4, "parameters": [
                             randi1(3, 97) + randi2(-1, 1)]},
            {"filter_id": 6, "parameters": [
                             randi1(42, 48) + randi2(-1, 1),
                             randi1(42, 48) + randi2(-1, 1),
                             randi1(42, 48) + randi2(-1, 1)]},
            ]


def _get_filter_config(args):
    # filter id 0: Exposure: increase or decrease the brightness of the image
    # filter id 1: Gamma: increase or decrease the gamma of the image
    # filter id 2: Saturation: increase or decrease the saturation of the image
    # filter id 4: Contrast: increase or decrease the saturation of the image
    # filter id 6: Tone: increase or decrease the tone (shadows, midtones, highlights) of the image
    seed1 = args.random_seed
    seed2 = 0 if '/cube_front/' in str(args.input_image_path) else 1
    random_config = get_random_mode_config(seed1, seed2)
    if args.mode == 'fixed':
        filter_config = [{"filter_id": 0, "parameters": [46]},
                         {"filter_id": 1, "parameters": [51]},
                         {"filter_id": 2, "parameters": [11]},
                         {"filter_id": 4, "parameters": [42]},
                         {"filter_id": 6, "parameters": [36, 56, 64]},
                         ]
    elif args.mode == 'random':
        filter_config = random_config
    elif args.mode == 'dark':
        filter_config = get_dark_mode_config(seed1, seed2)
    else:
        raise ValueError('Mode not supported.')
    return filter_config


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _main(args):
    assert os.path.isfile(args.input_image_path), \
        'Input image path does not exist.'
    input_path = args.input_image_path
    output_path = args.output_image_path
    if output_path is None or output_path in ("None", ""):
        output_path = Path(input_path).parent / \
            _append_filename(input_path, f'isp_{args.mode}')
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    filter_config = _get_filter_config(args)
    image = (cv2.imread(input_path)[:, :, ::-1] /
             255.0).astype(np.float32)
    image = _apply_filters(image, filter_config)
    cv2.imwrite(str(output_path), image[:, :, ::-1] * 255.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply ISP effects to an image.')
    parser.add_argument('-i', '--input-image-path', required=True,
                        help='path of the input image')
    parser.add_argument('-o', '--output-image-path',
                        help='Directory of the output images')
    parser.add_argument('-m', '--mode', default='fixed',
                        help='isp mode to use: fixed | random | dark')
    parser.add_argument('-s', '--random-seed', default=0, type=int,
                        help='seed for random or dark mode')
    args = parser.parse_args()
    _main(args)
