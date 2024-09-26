#!/usr/bin/env python3
import argparse
import distutils.spawn
import logging
import random
import re
import shutil

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import run

import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

MAX_UINT32 = 2**32 - 1


def _extract_integer_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


def _get_seed_from_filename(filename):
    filename_num = _extract_integer_from_filename(Path(filename).stem)
    return filename_num % MAX_UINT32


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _panorama_to_3pinholes(input_file, args):
    '''
    Input is a single image panorama filepath. Output is two 3pinhole image filepaths.
    '''
    assert (isinstance(input_file, Path))
    intermediate_dir = Path(args.output_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_3pinholes_exe = distutils.spawn.find_executable(
        "panorama_to_3pinholes.py")
    assert panorama_to_3pinholes_exe, 'Error: python executable `panorama_to_3pinholes.py` is not available!'
    cmds = [panorama_to_3pinholes_exe, f'--input-image-path={input_file}']
    output_filepath0 = intermediate_dir / _append_filename(input_file, 'down')
    output_filepath1 = intermediate_dir / _append_filename(input_file, 'up')
    cmds.extend(
        (
            f'--output-image-path0={str(output_filepath0)}',
            f'--output-image-path1={str(output_filepath1)}',
            f'--up-baseline={args.up_baseline}',
            f'--down-baseline={args.down_baseline}',
        )
    )

    if args.random_roll_max == 0.0:
        cmds.extend(
            (
                f'--roll={args.roll}',
                f'--pitch={args.pitch}',
                f'--yaw={args.yaw}',
            )
        )
    else:
        seed = _get_seed_from_filename(input_file)
        random.seed(seed)
        val = args.random_roll_max
        roll_one_decimal = round(random.uniform(-val, val), 1)
        cmds.append(f'--roll={roll_one_decimal}')

    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.depth:
        cmds.append('--depth')
    if args.segmentation_graymap:
        cmds.append('--segmentation-graymap')
    if args.delete_odd_rows:
        cmds.append('--delete-odd-rows')
    if args.panorama_to_3pinholes:
        cmds.append(
            f'--overwrite-remapping-filepath={args.panorama_to_3pinholes}')
    if args.output_depth:
        cmds.append('--output-depth')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return [output_filepath0, output_filepath1]


def _process_image(input_file, args):
    _panorama_to_3pinholes(input_file, args)


def _main(args):
    input_files = list(Path(args.input_dir).iterdir())
    assert input_files, 'Error: can\'t find the any image files!'
    input_files = np.sort(np.array(input_files))
    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.count is not None and 0 < args.count < len(input_files):
        if args.random_seed != 0:
            np.random.seed(args.random_seed)
            input_files = np.random.choice(
                input_files, args.count, replace=False)
        else:
            input_files = input_files[:args.count]
    # Process the first image and cache the remapping table
    if input_files.size > 0:
        _process_image(input_files[0], args)
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye pipeline: panorama -> 3pinholes')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory of the output images')
    parser.add_argument('--panorama-to-3pinholes',
                        help='path of the remapping table')
    parser.add_argument('--output-depth', action='store_true',
                        help='whether output depth or disparity image')
    parser.add_argument('-c', '--count', type=int, default=0,
                        help='number of images to generate. 0 means all images')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='a random seed to select |--count| of images. 0 means no randomization')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('--roll', type=float, default=0.0,
                        help='extra roll angle in degrees')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='extra pitch angle in degrees')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='extra yaw angle in degrees')
    parser.add_argument('--random-roll-max', type=float, default=0.0,
                        help='use a random roll angle in degrees')
    parser.add_argument('-ub', '--up-baseline', type=float, default=0.09,
                        help='up camera baseline')
    parser.add_argument('-db', '--down-baseline', type=float, default=0.105,
                        help='down camera baseline')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    parser.add_argument('--delete-odd-rows', action='store_true',
                        help='delete odd row of the remapped image')
    args = parser.parse_args()
    _main(args)
