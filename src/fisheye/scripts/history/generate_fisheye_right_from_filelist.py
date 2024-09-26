#!/usr/bin/env python3
'''
TODO
'''
from functools import partial
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import random
from subprocess import run
import argparse
import distutils.spawn
import logging
import numpy as np
import shutil
import time
import os
import re

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


kOutputDir = 'output'
kIntermediateDir = 'intermediate'
kNoiseMagnitude = 0.15


def _get_prefix_number(filepath):
    basename = os.path.basename(filepath)
    match = re.match(r'(\d+)', basename)
    assert match, f'Error: cannot find prefix number in {filepath}!'
    return int(match[0])


def _split_arrays(arr):
    divisible_by_80 = []
    not_divisible_by_80 = []
    for item in arr:
        prefix_num = _get_prefix_number(item[1])
        if prefix_num % 80 == 0:
            divisible_by_80.append(item)
        else:
            not_divisible_by_80.append(item)
    return divisible_by_80, not_divisible_by_80


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _panorama_to_fisheye(input_file, output_file, args, roll, pitch, yaw):
    '''
    Input is a single image panorama filepath. Output is two round fisheye image filepaths.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(output_file).parent
    folder_id = 0
    if roll is not None:
        folder_id += 1
    if pitch is not None:
        folder_id += 2
    if yaw is not None:
        folder_id += 4
    if folder_id == 0:
        output_dir = output_path / 'data'
    else:
        output_dir = output_path / f'data_noisy_{folder_id}'

    final_output_dir = output_dir
    output_dir = output_dir / kIntermediateDir / 'fisheye'
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        rpy_record = final_output_dir / 'rpy_record.json'
        if folder_id > 0 and not rpy_record.exists():
            with open(rpy_record, 'w') as f:
                rpy = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
                json.dump(rpy, f)
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
    if folder_id > 0:
        cmds.append(f'--roll={roll if roll is not None else 0.0}')
        cmds.append(f'--pitch={pitch if pitch is not None else 0.0}')
        cmds.append(f'--yaw={yaw if yaw is not None else 0.0}')
    cmds.append('--use-right-intrinsics')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return [output_filepath0, output_filepath1]


def _add_blur(input_file, args):
    '''
    Input is a single image filepath.
    Output is a single image filepath, where the image is blurred.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(input_file).parent
    intermediate_dir = output_path.parent / 'blur'
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


def _get_random_val(magnitude):
    return np.random.uniform(low=-magnitude, high=magnitude)


def _convert1to3(input_file, args):
    '''
    Input is a single round fisheye image filepath.
    Output is a single remapped 1to3 pinhole image filepath.
    '''
    assert (isinstance(input_file, Path))
    output_path = Path(input_file).parent.parent.parent

    fisheye_to_3pinholes_exe = distutils.spawn.find_executable(
        "fisheye_to_3pinholes.py")
    assert fisheye_to_3pinholes_exe, 'Error: executable `fisheye_to_3pinholes.py` is not available!'
    cmds = [fisheye_to_3pinholes_exe, f'--input-image-path={input_file}']
    output_filepath = output_path / input_file.name
    cmds.append(f'--output-image-path={str(output_filepath)}')
    cmds.append('--delete-odd-rows')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return output_filepath


def _process_image_rpy(input_output_pair, args, roll=None, pitch=None, yaw=None):
    input_file, output_file = input_output_pair
    intermediate_files = []
    fisheye_files = _panorama_to_fisheye(
        Path(input_file), output_file, args, roll, pitch, yaw)
    intermediate_files += fisheye_files
    for fisheye_file in fisheye_files:
        blurred_fisheye_file = _add_blur(fisheye_file, args)
        rectified_file = _convert1to3(blurred_fisheye_file, args)
        if rectified_file.is_file():
            _logger.info(
                f'Generated a rectified image file : {rectified_file}')
        intermediate_files.append(blurred_fisheye_file)
    if not args.keep_intermediate_files:
        for intermediate_file in intermediate_files:
            if intermediate_file.is_file():
                intermediate_file.unlink()


def _process_image_select_a_noise(input_output_pair, args):
    _process_image_rpy(input_output_pair, args)

    selected = random.randint(0, 3)
    random_roll = _get_random_val(kNoiseMagnitude)
    random_pitch = _get_random_val(kNoiseMagnitude)
    random_yaw = _get_random_val(kNoiseMagnitude)
    if selected == 0:
        _process_image_rpy(input_output_pair, args, roll=random_roll)
    elif selected == 1:
        _process_image_rpy(input_output_pair, args, pitch=random_pitch)
    elif selected == 2:
        _process_image_rpy(input_output_pair, args, yaw=random_yaw)
    else:
        _process_image_rpy(input_output_pair, args, random_roll,
                           random_pitch, random_yaw)


def _process_image_all_noises(input_output_pair, args):
    _process_image_rpy(input_output_pair, args)

    random_roll = _get_random_val(kNoiseMagnitude)
    random_pitch = _get_random_val(kNoiseMagnitude)
    random_yaw = _get_random_val(kNoiseMagnitude)
    _process_image_rpy(input_output_pair, args, roll=random_roll)
    _process_image_rpy(input_output_pair, args, pitch=random_pitch)
    _process_image_rpy(input_output_pair, args, yaw=random_yaw)

    random_roll = _get_random_val(kNoiseMagnitude)
    random_pitch = _get_random_val(kNoiseMagnitude)
    random_yaw = _get_random_val(kNoiseMagnitude)
    _process_image_rpy(input_output_pair, args, random_roll,
                       random_pitch, random_yaw)


def _main(args):
    filelist = np.load(args.input_filelist, allow_pickle=True)
    loaded_dict = filelist['rgb_right_file_mapping'].item()
    sorted_dict = dict(
        sorted(loaded_dict.items(), key=lambda item: _get_prefix_number(item[1])))
    input_output_pairs = list(sorted_dict.items())
    first_file_per_batch, other_files = _split_arrays(input_output_pairs)
    # Generate the first file per batch first to cache the remapping table.
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image_all_noises, args=args), first_file_per_batch)
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image_select_a_noise, args=args), other_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye pipelines')
    parser.add_argument('-i', '--input-filelist', required=True,
                        help='filelist of the input data')
    parser.add_argument('-k', '--keep-intermediate-files', action='store_true',
                        help='Whether to keep the intermediate fisheye images')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run')
    args = parser.parse_args()
    _main(args)
