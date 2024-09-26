#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np


# kRemappingTableFolder = '/media/autel/sim_ssd/4perception/low_altitude/remapping_table'
kRemappingTableFolder = '/mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table'


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
    return [left_p2f_path, left_p2p_path, f2p_path, right_p2f_noiseless, p2f_files]


def _save_right_p2f_noisy_files(p2f_files):
    p2f_noisy_files = [
        file for file in p2f_files if 'data_noisy_' in file
    ]
    right_p2f_file_to_rpy = {}
    for file in p2f_noisy_files:
        p = Path(file)
        folder = p.parent.parent.parent
        rpy = folder / 'rpy_record.json'
        with open(rpy, 'r') as f:
            data = json.load(f)
            data = {k: v for k, v in data.items() if v is not None}
        right_p2f_file_to_rpy[file] = data
    np.savez_compressed('right_p2f_file_to_rpy.npz',
                        right_p2f_file_to_rpy=right_p2f_file_to_rpy)


left_p2f_path, left_p2p_path, f2p_path, right_p2f_noiseless, p2f_files = _get_remapping_table(
    kRemappingTableFolder)
_save_right_p2f_noisy_files(p2f_files)

# right_p2f_file2rpy = np.load('right_p2f_file_to_rpy.npz', allow_pickle=True)[
#     'right_p2f_file_to_rpy'].item()
