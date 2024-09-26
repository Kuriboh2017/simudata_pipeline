#!/usr/bin/env python3
'''
TODO
'''
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import run
import argparse
import distutils.spawn
import logging
import numpy as np
import shutil
import time

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


kOutputDir = 'output'
kIntermediateDir = 'intermediate'


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _panorama_to_fisheye(input_file, output_file, args):
    '''
    Input is a single image panorama filepath. Output is two round fisheye image filepaths.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(output_file).parent
    output_dir = output_path / kIntermediateDir / 'fisheye'
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_fisheye_exe = distutils.spawn.find_executable(
        "panorama_to_fisheye.py")
    assert panorama_to_fisheye_exe, 'Error: executable `panorama_to_fisheye.py` is not available!'
    cmds = [panorama_to_fisheye_exe, f'--input-image-path={input_file}']
    output_filepath0 = output_dir / _append_filename(output_file, 'down')
    output_filepath1 = output_dir / _append_filename(output_file, 'up')
    cmds.append(
        f'--output-image-path0={str(output_filepath0)}')
    cmds.append(
        f'--output-image-path1={str(output_filepath1)}')
    cmds.append(
        f'--overwrite-remapping-filepath={str(args.panorama_to_fisheye)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return [output_filepath0, output_filepath1]


def _add_blur(input_file, output_file, args):
    '''
    Input is a single image filepath.
    Output is a single image filepath, where the image is blurred.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(output_file).parent
    intermediate_dir = output_path / kIntermediateDir / 'blur'
    if not args.dry_run:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    add_blur_exe = distutils.spawn.find_executable(
        "add_blur.py")
    assert add_blur_exe, 'Error: python executable `add_blur.py` is not available!'
    cmds = [add_blur_exe, f'--input-image-path={input_file}']
    output_filepath = intermediate_dir / input_file.name
    cmds.extend(
        (
            f'--output-image-path={str(output_filepath)}',
        )
    )
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return output_filepath


def _convert1to3(input_file, output_file, args):
    '''
    Input is a single round fisheye image filepath.
    Output is a single remapped 1to3 pinhole image filepath.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(output_file).parent

    fisheye_to_3pinholes_exe = distutils.spawn.find_executable(
        "fisheye_to_3pinholes.py")
    assert fisheye_to_3pinholes_exe, 'Error: executable `fisheye_to_3pinholes.py` is not available!'
    cmds = [fisheye_to_3pinholes_exe, f'--input-image-path={input_file}']
    output_filepath = output_path / input_file.name
    cmds.append(f'--output-image-path={str(output_filepath)}')
    cmds.append('--delete-odd-rows')
    cmds.append(
        f'--overwrite-remapping-filepath={str(args.fisheye_to_3pinholes)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return output_filepath


def _process_image(input_output_pair, args):
    input_file, output_file = input_output_pair
    intermediate_files = []
    fisheye_files = _panorama_to_fisheye(Path(input_file), output_file, args)
    intermediate_files += fisheye_files
    for fisheye_file in fisheye_files:
        blurred_fisheye_file = _add_blur(fisheye_file, output_file, args)
        rectified_file = _convert1to3(blurred_fisheye_file, output_file, args)
        if rectified_file.is_file():
            _logger.info(
                f'Generated a rectified image file : {rectified_file}')
        intermediate_files.append(blurred_fisheye_file)
    if not args.keep_intermediate_files:
        for intermediate_file in intermediate_files:
            if intermediate_file.is_file():
                intermediate_file.unlink()


def _main(args):
    filelist = np.load(args.input_filelist, allow_pickle=True)
    loaded_dict = filelist['rgb_left_file_mapping'].item()
    input_output_pairs = list(loaded_dict.items())
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_output_pairs)

    if not args.keep_intermediate_files:
        output_dirs = list(loaded_dict.values())
        for output_dir in output_dirs:
            intermediate_dir = Path(output_dir) / kIntermediateDir
            if intermediate_dir.is_dir():
                shutil.rmtree(intermediate_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye pipelines')
    parser.add_argument('-i', '--input-filelist', required=True,
                        help='filelist of the input data')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--panorama-to-fisheye', required=True,
                        help='remapping table filepath')
    parser.add_argument('--fisheye-to-3pinholes', required=True,
                        help='remapping table filepath')
    parser.add_argument('-k', '--keep-intermediate-files', action='store_true',
                        help='Whether to keep the intermediate fisheye images')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run')
    args = parser.parse_args()
    _main(args)
