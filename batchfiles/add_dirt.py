#!/usr/bin/env python3
import argparse
import distutils.spawn
import logging
import random
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import run

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

# Modify the random range here:
_BLUR_INTENSITY_RANGE = (1.0, 1.8)
_BLUR_KERNEL_SIZE_RANGE = (6, 12)
_BLUR_PSEUDO_OVEREXPOSURE_RANGE = (5, 20)
_DIRT_TEXTURE_OFFSET_MAX = 1024
_DIRT_TEXTURE_SCALE_RANGE = (0.3, 0.8)
_DIRT_TEXTURE_ID_RANGE = (0, 15)
_DIRT_TEXTURE_RATIO_RANGE = (0.1, 0.4)
# You can change the color scale range to (0,1) to check the dirt effects with other colors.
_DIRT_TEXTURE_RED_SCALE_RANGE = (0.9, 1.0)
_DIRT_TEXTURE_GREEN_SCALE_RANGE = (0.9, 1.0)
_DIRT_TEXTURE_BLUE_SCALE_RANGE = (0.9, 1.0)


def _process_image(input_file, args):
    '''
    Input is a single image filepath. The processed output is an image file with screen-dirt using the same filename.
    '''
    assert (isinstance(input_file, Path))
    screen_dirt_exe = distutils.spawn.find_executable("screen_dirt")
    assert screen_dirt_exe, 'Error: cpp executable `screen_dirt` is not available!'
    cmds = [screen_dirt_exe]
    cmds.append(f'--input-image-path={input_file}')
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    cmds.append(
        f'--output-image-path={Path.joinpath(output_folder, input_file.stem)}_screen_dirt{input_file.suffix}')
    seed = time.time()
    random.seed(seed)
    cmds.append(f'--blur-intensity={random.uniform(*_BLUR_INTENSITY_RANGE)}')
    cmds.append(
        f'--blur-kernel-size={random.uniform(*_BLUR_KERNEL_SIZE_RANGE)}')
    cmds.append(
        f'--blur-pseudo-overexposure={random.uniform(*_BLUR_PSEUDO_OVEREXPOSURE_RANGE)}')
    cmds.append(
        f'--dirt-texture-offset-x={random.randint(0, _DIRT_TEXTURE_OFFSET_MAX)}')
    cmds.append(
        f'--dirt-texture-offset-y={random.randint(0, _DIRT_TEXTURE_OFFSET_MAX)}')
    cmds.append(
        f'--dirt-texture-scale={random.uniform(*_DIRT_TEXTURE_SCALE_RANGE)}')
    cmds.append(f'--dirt-texture-id={random.randint(*_DIRT_TEXTURE_ID_RANGE)}')
    cmds.append(
        f'--dirt-texture-ratio={random.uniform(*_DIRT_TEXTURE_RATIO_RANGE)}')
    cmds.append(
        f'--dirt-texture-red-scale={random.uniform(*_DIRT_TEXTURE_RED_SCALE_RANGE)}')
    cmds.append(
        f'--dirt-texture-green-scale={random.uniform(*_DIRT_TEXTURE_GREEN_SCALE_RANGE)}')
    cmds.append(
        f'--dirt-texture-blue-scale={random.uniform(*_DIRT_TEXTURE_BLUE_SCALE_RANGE)}')
    _logger.info(f'Executing seed {seed} with command:\n{" ".join(cmds)}')
    run(cmds)


def _main(args):
    input_files = list(Path(args.input_dir).iterdir())
    assert input_files, 'Error: can\'t find the any image files!'
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add screen dirt to images.')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory of the output images')
    args = parser.parse_args()
    _main(args)
