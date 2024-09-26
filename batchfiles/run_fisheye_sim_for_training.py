#!/usr/bin/env python3
'''
Run fisheye simulation on a dataset for training.
Generate rpy noises for the right RGB camera only.
Optionally, add --blur to both left and right RGB cameras.
'''
from subprocess import run
from pathlib import Path
import logging
import distutils.spawn
import argparse
import json


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)
kJsonFilename = 'data.json'


def _update_folder_id(cam_dir, args):
    current_dir = Path(cam_dir)
    json_path = Path(current_dir) / kJsonFilename
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'highest_noisy_folder_id': 0}
    data['highest_noisy_folder_id'] += 1
    noisy_folder_id = data['highest_noisy_folder_id']
    data[noisy_folder_id] = {
        'roll': args.roll,
        'pitch': args.pitch,
        'yaw': args.yaw,
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    return noisy_folder_id


def _run_fisheye(input_dir, output_dir, subfolder, args, folder_id, seed, is_right_cam):
    input_path = input_dir / subfolder
    leaf_folder_name = f'{subfolder.name}_3to1'
    if args.noisy:
        leaf_folder_name = f'{leaf_folder_name}_noisy_{folder_id}'
    output_path = output_dir / subfolder.parent / leaf_folder_name
    run_fisheye_exe = distutils.spawn.find_executable(
        "run_fisheye.py")
    assert run_fisheye_exe, 'Error: python executable `run_fisheye.py` is not available!'
    cmds = [
        run_fisheye_exe,
        f'--input-dir={input_path}',
        f'--output-dir={output_path}',
        f'--count={args.count}',
        f'--random-seed={seed}',
        '--delete-odd-rows',
    ]
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.calibration_noise_level != 0.0:
        cmds.append(
            f'--calibration-noise-level={args.calibration_noise_level}')
    if args.blur:
        cmds.extend(
            ('--blur', f'--gaussian-kernel-size={args.blur_kernel_size}'))
    if args.keep_intermediate_images:
        cmds.append('--keep-intermediate-images')
    if is_right_cam:
        cmds.extend([f'--roll={args.roll}',
                     f'--pitch={args.pitch}',
                     f'--yaw={args.yaw}',
                     '--use-right-intrinsics'])
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _run_panorama_to_3pinholes(input_dir, output_dir, subfolder, args, seed, is_depth=False, is_segmentation_graymap=False):
    input_path = input_dir / subfolder
    if subfolder.name == 'Depth':
        subfolder = subfolder.parent / 'Disp'
    leaf_folder_name = f'{subfolder.name}_3to1'
    output_path = output_dir / subfolder.parent / leaf_folder_name

    run_panorama_to_3pinholes_exe = distutils.spawn.find_executable(
        "run_panorama_to_3pinholes.py")
    assert run_panorama_to_3pinholes_exe, 'Error: python executable `run_panorama_to_3pinholes.py` is not available!'
    cmds = [
        run_panorama_to_3pinholes_exe,
        f'--input-dir={input_path}',
        f'--output-dir={output_path}',
        f'--count={args.count}',
        f'--random-seed={seed}',
        f'--delete-odd-rows',
    ]
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.keep_intermediate_images:
        cmds.append('--keep-intermediate-images')
    if is_depth:
        cmds.append('--depth')
    if is_segmentation_graymap:
        cmds.append('--segmentation-graymap')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / input_dir.name
    if args.roll != 0.0 or args.pitch != 0.0 or args.yaw != 0.0:
        assert args.noisy, 'Error: rotation is only supported for noisy images'
    fisheye_folder = output_dir / 'group1'
    fisheye_folder.mkdir(parents=True, exist_ok=True)
    folder_id = _update_folder_id(fisheye_folder, args) if args.noisy else None

    seed = args.random_seed

    # Group1 represents the fisheye camera group
    left_rgba_folder = Path('group1') / 'cam1_0' / 'Image'
    right_rgba_folder = Path('group1') / 'cam1_1' / 'Image'
    depth_folder = Path('group1') / 'cam1_0' / 'Depth'
    segmentation_color_folder = Path(
        'group1') / 'cam1_0' / 'Segmentation' / 'Colormap'
    segmentation_graymap_folder = Path(
        'group1') / 'cam1_0' / 'Segmentation' / 'Graymap'

    if not args.noisy:
        _run_fisheye(input_dir, output_dir, left_rgba_folder, args, folder_id, seed,
                     is_right_cam=False)
        _run_panorama_to_3pinholes(
            input_dir, output_dir, depth_folder, args, seed, is_depth=True)
        _run_panorama_to_3pinholes(
            input_dir, output_dir, segmentation_graymap_folder, args, seed, is_segmentation_graymap=True)
        _run_panorama_to_3pinholes(
            input_dir, output_dir, segmentation_color_folder, args, seed)

    _run_fisheye(input_dir, output_dir, right_rgba_folder, args, folder_id, seed,
                 is_right_cam=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye rgba pipeline: panorama -> fisheye -> 3pinholes')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the sim dataset')
    parser.add_argument('-o', '--output-dir',
                        help='Output directory')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('-k', '--keep-intermediate-images', action='store_true',
                        help='Whether to keep the intermediate fisheye images')
    parser.add_argument('--roll', type=float, default=0.0,
                        help='extra roll angle in degrees')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='extra pitch angle in degrees')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='extra yaw angle in degrees')
    parser.add_argument('--noisy', action='store_true',
                        help='create a noisy folder')
    parser.add_argument('--blur', action='store_true',
                        help='Whether to add the gaussian blur effects')
    parser.add_argument('--blur-kernel-size', default=3,
                        type=int, help='Gaussian blur kernel size')
    parser.add_argument('-l', '--calibration-noise-level', type=float, default=0.0,
                        help='noise level of the camera intrinsic parameters')
    parser.add_argument('-c', '--count', type=int, default=0,
                        help='number of images to generate. 0 means all images')
    parser.add_argument('-seed', '--random-seed', type=int, default=0,
                        help='a random seed to select |--count| of images. 0 means no randomization')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='whether to do a dry run')
    args = parser.parse_args()
    _main(args)
