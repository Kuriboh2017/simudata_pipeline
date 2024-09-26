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
_SUN_SIZE_RANGE = (0.12, 0.16)
_SUN_LOCATION_RANGE = (-1.0, 1.0)
_DIFFRACTION_SPIKES_INTENSITY_RANGE = (0.8, 3.0)
_LENS_FLARES_COUNT_RANGE = (15, 22)
_LENS_FLARES_INTENSITY_RANGE = (0.8, 1.2)


def _process_image(input_file, args):
    '''
    Input is a single image filepath. The processed output is an image file with lens flare using the same filename.
    '''
    assert (isinstance(input_file, Path))
    lens_flare_exe = distutils.spawn.find_executable("lens_flare")
    assert lens_flare_exe, 'Error: cpp executable `lens_flare` is not available!'
    cmds = [lens_flare_exe]
    cmds.append(f'--input-image-path={input_file}')
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    cmds.append(
        f'--output-image-path={Path.joinpath(output_folder, input_file.stem)}_lens_flare{input_file.suffix}')
    seed = time.time()
    random.seed(seed)
    cmds.append(f'--sun-size={random.uniform(*_SUN_SIZE_RANGE)}')
    cmds.append(f'--sun-location-x={random.uniform(*_SUN_LOCATION_RANGE)}')
    cmds.append(f'--sun-location-y={random.uniform(*_SUN_LOCATION_RANGE)}')
    cmds.append(
        f'--diffraction-spikes-intensity={random.uniform(*_DIFFRACTION_SPIKES_INTENSITY_RANGE)}')
    cmds.append(
        f'--lens-flares-count={random.randint(*_LENS_FLARES_COUNT_RANGE)}')
    cmds.append(f'--lens-flares-seed={seed}')
    cmds.append(
        f'--lens-flares-intensity={random.uniform(*_LENS_FLARES_INTENSITY_RANGE)}')
    run(cmds)


def _main(args):
    input_files = list(Path(args.input_dir).iterdir())
    assert input_files, 'Error: can\'t find the any image files!'
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add lens flare to images.')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory of the output images')
    args = parser.parse_args()
    _main(args)
