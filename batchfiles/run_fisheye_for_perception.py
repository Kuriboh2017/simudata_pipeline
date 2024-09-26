#!/usr/bin/env python3
'''
Run fisheye simulation on a dataset for training.
Don't add any noises on left cameras.
Add rpy noises for the right RGB camera only.
'''
from subprocess import run
from pathlib import Path
import logging
import distutils.spawn
import argparse


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)
kJsonFilename = 'data.json'


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


def _run_fisheye(input_dir, output_dir, subfolder, args, panorama_to_fisheye, fish_to_3pinholes, seed, is_right_cam):
    input_path = input_dir / subfolder
    leaf_folder_name = f'{subfolder.name}_3to1'
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
    cmds.append('--blur')
    if args.keep_intermediate_images:
        cmds.append('--keep-intermediate-images')
    cmds.append(f'--fisheye-to-3pinholes={fish_to_3pinholes}')
    if is_right_cam:
        cmds.append('--use-right-intrinsics')
    else:
        cmds.append(f'--panorama-to-fisheye={panorama_to_fisheye}')
    if args.remapping_folder:
        cmds.append(f'--remapping-folder={args.remapping_folder}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _run_panorama_to_3pinholes(input_dir, output_dir, subfolder, args, p2p_path, seed, is_depth=False, is_segmentation_graymap=False):
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
    cmds.append(f'--panorama-to-3pinholes={p2p_path}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / input_dir.name
    fisheye_folder = output_dir / 'group1'
    fisheye_folder.mkdir(parents=True, exist_ok=True)

    left_p2f_path, left_p2p_path, f2p_path, _ = _get_remapping_table(
        args.remapping_folder)

    seed = args.random_seed

    # Group1 represents the fisheye camera group
    left_rgba_folder = Path('group1') / 'cam1_0' / 'Image'
    right_rgba_folder = Path('group1') / 'cam1_1' / 'Image'
    depth_folder = Path('group1') / 'cam1_0' / 'Depth'
    segmentation_color_folder = Path(
        'group1') / 'cam1_0' / 'Segmentation' / 'Colormap'
    segmentation_graymap_folder = Path(
        'group1') / 'cam1_0' / 'Segmentation' / 'Graymap'

    _run_fisheye(input_dir, output_dir, left_rgba_folder, args, left_p2f_path, f2p_path, seed,
                 is_right_cam=False)
    _run_fisheye(input_dir, output_dir, right_rgba_folder, args, left_p2f_path, f2p_path, seed,
                 is_right_cam=True)
    _run_panorama_to_3pinholes(
        input_dir, output_dir, depth_folder, args, left_p2p_path, seed, is_depth=True)
    _run_panorama_to_3pinholes(
        input_dir, output_dir, segmentation_graymap_folder, args, left_p2p_path, seed, is_segmentation_graymap=True)
    _run_panorama_to_3pinholes(
        input_dir, output_dir, segmentation_color_folder, args, left_p2p_path, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye rgba pipeline: panorama -> fisheye -> 3pinholes')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the sim dataset')
    parser.add_argument('-o', '--output-dir',
                        help='Output directory')
    parser.add_argument('--remapping-folder', required=True,
                        help='Remapping folder')
    parser.add_argument('-r', '--rectify-config',
                        help='path of the rectification config file')
    parser.add_argument('-k', '--keep-intermediate-images', action='store_true',
                        help='Whether to keep the intermediate fisheye images')
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
