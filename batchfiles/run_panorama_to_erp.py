#!/usr/bin/env python3
import argparse
import distutils.spawn
import logging
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


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _panorama_to_erp(input_file, args):
    '''
    Input is a single image panorama filepath. Output is two erp image filepaths.
    '''
    assert (isinstance(input_file, Path))
    intermediate_dir = Path(args.output_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_erp_exe = distutils.spawn.find_executable(
        "panorama_to_erp.py")
    assert panorama_to_erp_exe, 'Error: python executable `panorama_to_erp.py` is not available!'
    cmds = [panorama_to_erp_exe, f'--input-image-path={input_file}']
    output_filepath0 = intermediate_dir / _append_filename(input_file, 'down')
    output_filepath1 = intermediate_dir / _append_filename(input_file, 'up')
    cmds.extend(
        (
            f'--output-image-path0={str(output_filepath0)}',
            f'--output-image-path1={str(output_filepath1)}',
        )
    )
    if args.disparity:
        cmds.append('--disparity')
    if args.segmentation_graymap:
        cmds.append('--segmentation-graymap')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return [output_filepath0, output_filepath1]


def _process_image(input_file, args):
    _panorama_to_erp(input_file, args)


def _main(args):
    input_files = list(Path(args.input_dir).iterdir())
    assert input_files, 'Error: can\'t find the any image files!'
    input_files = np.sort(np.array(input_files))
    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye depth pipeline: panorama -> erp')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory of the output images')
    parser.add_argument('-d', '--disparity', action='store_true',
                        help='whether or not processing disparity image')
    parser.add_argument('-g', '--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    args = parser.parse_args()
    _main(args)
