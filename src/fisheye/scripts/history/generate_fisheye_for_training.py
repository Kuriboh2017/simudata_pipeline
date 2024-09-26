#!/usr/bin/env python3
'''
Run |run_fisheye_sim_for_training.py| on a dataset for training with different rpy noises.
Optionally, add --blur to both left and right RGB cameras.
'''
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


def _get_random_seed(magnitude):
    return np.random.uniform(low=-magnitude, high=magnitude)


def _run(args, roll=None, pitch=None, yaw=None):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    exe = distutils.spawn.find_executable(
        "run_fisheye_sim_for_training.py")
    assert exe, 'Error: python executable `run_fisheye_sim_for_training.py` is not available!'
    cmds = [
        exe,
        f'--input-dir={input_path}',
        f'--output-dir={output_path}',
    ]
    if roll is not None:
        cmds.append(f'--roll={roll}')
    if pitch is not None:
        cmds.append(f'--pitch={pitch}')
    if yaw is not None:
        cmds.append(f'--yaw={yaw}')
    if roll is not None or pitch is not None or yaw is not None:
        cmds.append('--noisy')
    if args.blur:
        cmds.append('--blur')
    if args.dry_run:
        cmds.append('--dry-run')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    run(cmds)


def _run_with_noise(args):
    # Noisy all of rpy
    random_roll = _get_random_seed(args.rpy_noise_magnitude)
    random_pitch = _get_random_seed(args.rpy_noise_magnitude)
    random_yaw = _get_random_seed(args.rpy_noise_magnitude)
    _run(args, roll=random_roll, pitch=random_pitch, yaw=random_yaw)

    # Noisy individual rpy
    rand_roll = _get_random_seed(args.rpy_noise_magnitude)
    rand_pitch = _get_random_seed(args.rpy_noise_magnitude)
    rand_yaw = _get_random_seed(args.rpy_noise_magnitude)
    _run(args, roll=rand_roll)
    _run(args, pitch=rand_pitch)
    _run(args, yaw=rand_yaw)


def _main(args):
    current_time = time.time()
    np.random.seed(int(current_time))

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_dir = output_root / input_dir.name
    assert input_dir.exists(), f'Error: {input_dir} does not exist!'
    shutil.rmtree(str(output_dir), ignore_errors=True)

    # Noiseless
    _run(args)

    if args.noisy:
        _run_with_noise(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye pipelines')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the sim dataset')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--rpy-noise-magnitude', type=float, default=0.15,
                        help='Magnitude of the noise for roll, pitch, yaw')
    parser.add_argument('--blur', action='store_true',
                        help='Whether to add the gaussian blur effects')
    parser.add_argument('--noisy', action='store_true',
                        help='Whether to add the rpy noise')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run')
    args = parser.parse_args()
    _main(args)
