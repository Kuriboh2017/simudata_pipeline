#!/usr/bin/env python3
import argparse
import distutils.spawn
import json
import logging
import random
import re
import numpy as np
import shutil

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import run

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

kOutputDir = 'output'
kIntermediateDir = 'intermediate'
kRightTableToRPY = 'right_p2f_file_to_rpy.npz'
MAX_UINT32 = 2**32 - 1


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def overwrite_shorter_suffix(filename, new_suffix='.json'):
    path = Path(filename)
    folder = path.parent.parent.parent
    path = folder / path.name
    new_path = path.with_suffix(new_suffix)
    return str(new_path)


def _get_remapping_table(remapping_folder):
    p = Path(remapping_folder)
    left_p2f_path = p / 'left' / \
        'remapping_table_panorama2fisheye.npz'
    left_p2p_path = p / 'left' / \
        'remapping_table_panorama2pinholes.npz'
    f2p_path = p / 'left' / \
        'remapping_table_fisheye2pinholes.npz'
    right_path = p / 'right'
    p2f_files = [
        str(file) for file in right_path.rglob('remapping_table_panorama2fisheye.npz')
    ]
    p2f_files.sort()
    right_p2f_noiseless = p2f_files[0]
    return [left_p2f_path, left_p2p_path, f2p_path, right_p2f_noiseless]


def _panorama_to_fisheye(input_file, args, scale_factor=None):
    '''
    Input is a single image panorama filepath. Output is two round fisheye image filepaths.
    '''
    assert (isinstance(input_file, Path))
    output_dir = args.fisheye_dir
    if output_dir is None or output_dir == 'None':
        output_dir = Path(args.output_dir) / kIntermediateDir / 'fisheye'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_fisheye_exe = distutils.spawn.find_executable(
        "panorama_to_fisheye.py")
    assert panorama_to_fisheye_exe, 'Error: executable `panorama_to_fisheye.py` is not available!'
    cmds = [panorama_to_fisheye_exe, f'--input-image-path={input_file}']
    output_filepath0 = output_dir / _append_filename(input_file, 'down')
    output_filepath1 = output_dir / _append_filename(input_file, 'up')
    cmds.append(
        f'--output-image-path0={str(output_filepath0)}')
    cmds.append(
        f'--output-image-path1={str(output_filepath1)}')
    cmds.extend(
        (
            f'--roll={args.roll}',
            f'--pitch={args.pitch}',
            f'--yaw={args.yaw}',
        )
    )
    if scale_factor:
        cmds.append(f'--fxfy-scale={scale_factor}')
    if args.depth:
        cmds.append('--depth')
    if args.use_right_intrinsics:
        cmds.append('--use-right-intrinsics')
        _, _, f2p_path, right_p2f_noiseless = _get_remapping_table(
            args.remapping_folder)
        if add_noise := random.choice([True, False]) and args.noisy:
            if args.regenerate_noise:
                _regenerate_noise_args(cmds, output_filepath0)
            elif args.remapping_folder:
                _add_noise_args(
                    args, cmds, output_filepath0, output_filepath1
                )
        else:
            cmds.append(
                f'--overwrite-remapping-filepath={str(right_p2f_noiseless)}')
    elif args.panorama_to_fisheye:
        cmds.append(
            f'--overwrite-remapping-filepath={str(args.panorama_to_fisheye)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return [output_filepath0, output_filepath1]


def _add_noise_args(args, cmds, output_filepath0, output_filepath1):
    path = Path(args.remapping_folder) / kRightTableToRPY
    right_p2f_file2rpy = np.load(path, allow_pickle=True)[
        'right_p2f_file_to_rpy'].item()
    random_remapping_table = random.choice(
        list(right_p2f_file2rpy.keys()))
    cmds.append(
        f'--overwrite-remapping-filepath={str(random_remapping_table)}')
    with open(overwrite_shorter_suffix(output_filepath0), 'w') as f:
        json.dump(right_p2f_file2rpy[random_remapping_table], f)


def _regenerate_noise_args(cmds, output_filepath0):
    roll = random.uniform(-0.1, 0.1)
    pitch = random.uniform(-0.1, 0.1)
    yaw = random.uniform(-0.1, 0.1)
    number = random.randint(1, 4)
    if number == 1:
        noises = {'roll': roll}
        cmds.append(f'--roll={roll}')
    elif number == 2:
        noises = {'pitch': pitch}
        cmds.append(f'--pitch={pitch}')
    elif number == 3:
        noises = {'yaw': yaw}
        cmds.append(f'--yaw={yaw}')
    else:
        noises = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        cmds.append(f'--roll={roll}')
        cmds.append(f'--pitch={pitch}')
        cmds.append(f'--yaw={yaw}')
    cmds.append('--recalculate')
    with open(overwrite_shorter_suffix(output_filepath0), 'w') as f:
        json.dump(noises, f)


def _extract_integer_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


def _get_seed_from_filename(filename):
    filename_num = _extract_integer_from_filename(Path(filename).stem)
    return filename_num % MAX_UINT32


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _add_blur(input_file, args):
    '''
    Input is a single image filepath.
    Output is a single image filepath, where the image is blurred.
    '''
    assert (isinstance(input_file, Path))
    assert args.gaussian_kernel_size % 2 == 1, "Kernel size must be odd"
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'blur'
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    add_blur_exe = distutils.spawn.find_executable(
        "add_blur.py")
    assert add_blur_exe, 'Error: python executable `add_blur.py` is not available!'
    cmds = [add_blur_exe, f'--input-image-path={input_file}']
    output_filepath = intermediate_dir / input_file.name
    cmds.append(f'--output-image-path={str(output_filepath)}')
    cmds.append(f'--gaussian-kernel-size={args.gaussian_kernel_size}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _add_vignette(fisheye_file, args):
    action_exe = distutils.spawn.find_executable('add_vignette.py')
    assert action_exe, 'Error: python executable `add_vignette.py` is not available!'
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'vignette'
    cmds = [action_exe, f'--input-image-path={fisheye_file}']
    output_filepath = intermediate_dir / \
        _append_filename(fisheye_file.name, 'vignette')
    cmds.append(f'--output-image-path={str(output_filepath)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _add_demosaicing(fisheye_file, args):
    action_exe = distutils.spawn.find_executable('add_demosaicing.py')
    assert action_exe, 'Error: python executable `add_demosaicing.py` is not available!'
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'demosaicing'
    cmds = [action_exe, f'--input-image-path={fisheye_file}']
    output_filepath = intermediate_dir / \
        _append_filename(fisheye_file.name, 'demosaicing')
    cmds.append(f'--output-image-path={str(output_filepath)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _add_fixed_isp(fisheye_file, args):
    action_exe = distutils.spawn.find_executable('add_isp.py')
    assert action_exe, 'Error: python executable `add_isp.py` is not available!'
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'fixed_isp'
    cmds = [action_exe, f'--input-image-path={fisheye_file}']
    output_filepath = intermediate_dir / \
        _append_filename(fisheye_file.name, 'fixed_isp')
    cmds.extend(
        (f'--output-image-path={str(output_filepath)}', '--mode=fixed'))
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _add_random_isp(fisheye_file, args):
    action_exe = distutils.spawn.find_executable('add_isp.py')
    assert action_exe, 'Error: python executable `add_isp.py` is not available!'
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'random_isp'
    cmds = [action_exe, f'--input-image-path={fisheye_file}']
    output_filepath = intermediate_dir / \
        _append_filename(fisheye_file.name, 'random_isp')
    cmds.extend(
        (f'--output-image-path={str(output_filepath)}', '--mode=random'))
    seed = _get_seed_from_filename(fisheye_file)
    cmds.append(f'--seed={seed}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _add_dark_isp(fisheye_file, args):
    action_exe = distutils.spawn.find_executable('add_isp.py')
    assert action_exe, 'Error: python executable `add_isp.py` is not available!'
    intermediate_dir = Path(args.output_dir) / kIntermediateDir / 'dark_isp'
    cmds = [action_exe, f'--input-image-path={fisheye_file}']
    output_filepath = intermediate_dir / \
        _append_filename(fisheye_file.name, 'dark_isp')
    cmds.extend((f'--output-image-path={str(output_filepath)}', '--mode=dark'))
    seed = _get_seed_from_filename(fisheye_file)
    cmds.append(f'--seed={seed}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _convert1to3(input_file, args):
    '''
    Input is a single round fisheye image filepath.
    Output is a single remapped 1to3 pinhole image filepath.
    '''
    assert (isinstance(input_file, Path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fisheye_to_3pinholes_exe = distutils.spawn.find_executable(
        "fisheye_to_3pinholes.py")
    assert fisheye_to_3pinholes_exe, 'Error: executable `fisheye_to_3pinholes.py` is not available!'
    cmds = [fisheye_to_3pinholes_exe, f'--input-image-path={input_file}']
    output_filepath = output_dir / input_file.name
    cmds.append(f'--output-image-path={str(output_filepath)}')
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.calibration_noise_level != 0.0:
        cmds.append(
            f'--calibration-noise-level={args.calibration_noise_level}')
    if args.use_right_intrinsics:
        cmds.append('--use-right-intrinsics')
    if args.delete_odd_rows:
        cmds.append('--delete-odd-rows')
    if args.depth:
        cmds.extend(('--depth', '--output-depth'))
    if args.fisheye_to_3pinholes:
        cmds.append(
            f'--overwrite-remapping-filepath={str(args.fisheye_to_3pinholes)}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)
    return output_filepath


def _process_fisheyes(fisheye_files, args):
    for fisheye_file in fisheye_files:
        if args.blur:
            fisheye_file = _add_blur(fisheye_file, args)
        if args.vignette:
            fisheye_file = _add_vignette(fisheye_file, args)
        if args.demosaicing:
            fisheye_file = _add_demosaicing(fisheye_file, args)
        if args.fixed_isp:
            fisheye_file = _add_fixed_isp(fisheye_file, args)
        if args.random_isp:
            fisheye_file = _add_random_isp(fisheye_file, args)
        if args.dark_isp:
            fisheye_file = _add_dark_isp(fisheye_file, args)
        rectified_file = _convert1to3(fisheye_file, args)
        if rectified_file.is_file():
            _logger.info(
                f'Generated a rectified image file : {rectified_file}')


def _process_image(input_file, args):
    fisheye_files = _panorama_to_fisheye(input_file, args)
    _process_fisheyes(fisheye_files, args)


def _process_image_scale(input_file_scale, args):
    input_file, scale_factor = input_file_scale
    fisheye_files = _panorama_to_fisheye(input_file, args, scale_factor)
    _process_fisheyes(fisheye_files, args)


def exec_with_scale(input_files, args):
    num_of_files = input_files.size
    begin = np.array([1.0] * int(num_of_files * 0.25))
    mid = np.linspace(1.0, 1.01, int(num_of_files * 0.5) + 1)
    end = np.array([1.01] * (num_of_files - begin.size - mid.size))
    scale_factors = np.concatenate((begin, mid, end))
    input_file_scale = tuple(zip(input_files, scale_factors))
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image_scale, args=args), input_file_scale)


def run_fisheye_rgba(args):
    input_files = list(Path(args.input_dir).iterdir())
    assert input_files, 'Error: can\'t find the any image files!'
    input_files = np.array(
        sorted(input_files, key=lambda f: int(f.name.split('.')[0])))
    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True)
    if args.count is not None and 0 < args.count < len(input_files):
        if args.random_seed != 0:
            np.random.seed(args.random_seed)
            input_files = np.random.choice(
                input_files, args.count, replace=False)
        else:
            input_files = input_files[:args.count]
    if args.scale:
        exec_with_scale(input_files, args)
    else:
        # Process the first image and cache the remapping table
        if input_files.size > 0:
            _process_image(input_files[0], args)
        with Pool(cpu_count()) as p:
            p.map(partial(_process_image, args=args), input_files[1:])
    if not args.keep_intermediate_images:
        intermediate_dir = Path(args.output_dir) / kIntermediateDir
        shutil.rmtree(intermediate_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye rgba pipeline: panorama -> fisheye -> rectify')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-f', '--fisheye-dir',
                        help='Directory of the fisheye images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory of the output images')
    parser.add_argument('--panorama-to-fisheye',
                        help='remapping table filepath')
    parser.add_argument('--fisheye-to-3pinholes',
                        help='remapping table filepath')
    parser.add_argument('--remapping-folder',
                        help='Remapping folder')
    parser.add_argument('-c', '--count', type=int, default=0,
                        help='number of images to generate. 0 means all images')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='a random seed to select |--count| of images. 0 means no randomization')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('--noisy', action='store_true',
                        help='create a noisy folder')
    parser.add_argument('--regenerate-noise', action='store_true',
                        help='regenerate noise')
    parser.add_argument('--scale', action='store_true',
                        help='scale camera intrinsics')
    parser.add_argument('--use-right-intrinsics', action='store_true',
                        help='use right intrinsics instead of left intrinsics')
    parser.add_argument('-k', '--keep-intermediate-images', action='store_true',
                        help='Whether to keep the intermediate fisheye images')
    parser.add_argument('--blur', action='store_true',
                        help='Whether to add the gaussian blur effects')
    parser.add_argument('--gaussian-kernel-size', default=3,
                        type=int, help='Gaussian blur kernel size')
    parser.add_argument('--demosaicing', action='store_true',
                        help='Whether to add the demosaicing effects')
    parser.add_argument('--vignette', action='store_true',
                        help='Whether to add the vignette effects')
    parser.add_argument('--fixed-isp', action='store_true',
                        help='Whether to add the fixed-isp effects')
    parser.add_argument('--random-isp', action='store_true',
                        help='Whether to add the random-isp effects')
    parser.add_argument('--dark-isp', action='store_true',
                        help='Whether to add the dark-isp effects')
    parser.add_argument('-l', '--calibration-noise-level', type=float, default=0.0,
                        help='noise level of the camera intrinsic parameters')
    parser.add_argument('--roll', type=float, default=0.0,
                        help='extra roll angle in degrees')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='extra pitch angle in degrees')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='extra yaw angle in degrees')
    parser.add_argument('--delete-odd-rows', action='store_true',
                        help='delete odd row of the remapped image')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    args = parser.parse_args()
    run_fisheye_rgba(args)
