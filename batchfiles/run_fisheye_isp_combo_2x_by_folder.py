#!/usr/bin/env python3
'''
Run fisheye simulation on a dataset for training.
Generate rpy noises for the right RGB camera only.

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


def _run_fisheye(input_dir, output_dir, subfolder, args, panorama_to_fisheye, fish_to_3pinholes):
    import run_fisheye_isp_with_erp
    input_path = input_dir / subfolder
    leaf_folder_name = f'{subfolder.name}_3to1'
    output_path = output_dir / subfolder.parent / leaf_folder_name
    
    
    '''
    cmd控制的版本
    run_fisheye_exe = distutils.spawn.find_executable(
        "run_fisheye_isp_with_erp.py")
    assert run_fisheye_exe, 'Error: python executable `run_fisheye_isp_with_erp.py` is not available!'
    cmds = [
        run_fisheye_exe,
        f'--input-dir={input_path}',
        f'--output-dir={output_path}',
        f'--count={args.count}',
    ]
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.calibration_noise_level != 0.0:
        cmds.append(
            f'--calibration-noise-level={args.calibration_noise_level}')
    if args.keep_intermediate_images:
        cmds.append('--keep-intermediate-images')
    cmds.append(f'--fisheye-to-3pinholes={fish_to_3pinholes}')
    cmds.append(f'--panorama-to-fisheye={panorama_to_fisheye}')
    if args.remapping_folder:
        cmds.append(f'--remapping-folder={args.remapping_folder}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    '''


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
    ]
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    if args.keep_intermediate_images:
        cmds.append('--keep-intermediate-images')
    if is_depth:
        cmds.append('--depth')
        cmds.append('--output-depth')
    if is_segmentation_graymap:
        cmds.append('--segmentation-graymap')
    cmds.append(f'--panorama-to-3pinholes={p2p_path}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _run_panorama_to_erp(input_dir, output_dir, subfolder, args, is_disparity=False, is_segmentation_graymap=False):
    input_path = input_dir / subfolder
    leaf_folder_name = f'{subfolder.name}_erp'
    output_path = output_dir / subfolder.parent / leaf_folder_name

    run_panorama_to_erp_exe = distutils.spawn.find_executable(
        "run_panorama_to_erp.py")
    assert run_panorama_to_erp_exe, 'Error: python executable `run_panorama_to_erp.py` is not available!'
    cmds = [
        run_panorama_to_erp_exe,
        f'--input-dir={input_path}',
        f'--output-dir={output_path}',
    ]
    if is_disparity:
        cmds.append('--disparity')
    if is_segmentation_graymap:
        cmds.append('--segmentation-graymap')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # up_baseline = 0.09
    # if 'MX128' in str(input_dir):
    #     up_baseline = 0.128

    left_p2f_path, left_p2p_path, f2p_path, _ = _get_remapping_table(
        args.remapping_folder)

    seed = args.random_seed

    # Group1 represents the fisheye camera group
    rgba_folders = [Path('cube_front') / 'CubeScene',
                    Path('cube_below') / 'CubeScene',
                    Path('cube_rear') / 'CubeScene',]
    depth_folder = Path('cube_front') / 'CubeDepth'
    disparity_folder_up = Path('cube_front') / \
        f'CubeDisparity_{str(args.up_baseline)}'
    disparity_folder_down = Path('cube_front') / \
        f'CubeDisparity_{str(args.down_baseline)}'
    segmentation_color_folder = Path(
        'cube_front') / 'CubeSegmentation' / 'Colormap'
    segmentation_graymap_folder = Path(
        'cube_front') / 'CubeSegmentation' / 'Graymap'

    for rgba_folder in rgba_folders:
        _run_fisheye(input_dir, output_dir, rgba_folder,
                     args, left_p2f_path, f2p_path)
    # _run_panorama_to_3pinholes(
    #     input_dir, output_dir, depth_folder, args, left_p2p_path, seed, is_depth=True)
    # _run_panorama_to_3pinholes(
    #     input_dir, output_dir, segmentation_color_folder, args, left_p2p_path, seed)
    # _run_panorama_to_3pinholes(
    #     input_dir, output_dir, segmentation_graymap_folder, args, left_p2p_path, seed, is_segmentation_graymap=True)

    _run_panorama_to_erp(input_dir, output_dir,
                         disparity_folder_up, args, is_disparity=True)
    _run_panorama_to_erp(input_dir, output_dir,
                         disparity_folder_down, args, is_disparity=True)
    _run_panorama_to_erp(
        input_dir, output_dir, segmentation_color_folder, args, is_segmentation_graymap=False)
    _run_panorama_to_erp(
        input_dir, output_dir, segmentation_graymap_folder, args, is_segmentation_graymap=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye rgba pipeline: panorama -> fisheye -> effect -> 3pinholes')
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