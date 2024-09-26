#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging
import numpy as np
import os

from lz4.frame import compress as lzcompress
from lz4.frame import decompress as lzdecompress
import pickle as pkl

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


# Load old format: 2x/ folder
def load_pkl_imgs(img_path):
    with open(img_path, "rb") as f_img_in:
        img_data = pkl.load(f_img_in)
        limg = np.frombuffer(lzdecompress(
            img_data['left_image']), dtype=img_data['image_dtype'])
        limg = limg.reshape(img_data['image_shape'])
        rimg = np.frombuffer(lzdecompress(
            img_data['right_image']), dtype=img_data['image_dtype'])
        rimg = rimg.reshape(img_data['image_shape'])
        left_img_path = img_data['left_path']
    return limg, rimg, left_img_path


def load_pkl_disp(disp_path):
    with open(disp_path, "rb") as f_disp_in:
        disp_data = pkl.load(f_disp_in)
        disp = np.frombuffer(lzdecompress(
            disp_data['left_disparity']), dtype=disp_data['disparity_dtype'])
        disp = disp.reshape(disp_data['disparity_shape'])
    return disp


def load_pkl_segs(seg_path):
    with open(seg_path, "rb") as f_seg_in:
        seg_data = pkl.load(f_seg_in)
        seg = np.frombuffer(lzdecompress(
            seg_data['segmentation']), dtype=seg_data['segment_dtype'])
        seg = seg.reshape(seg_data['segment_shape'])
    return seg


def load_mask_by_name(mask_data, name):
    tiny = mask_data[name]
    result = np.frombuffer(lzdecompress(tiny['data']), dtype=tiny['dtype'])
    result = result.reshape(tiny['shape'])
    return result


def load_pkl_masks(mask_path):
    with open(mask_path, "rb") as f_mask_in:
        mask_data = pkl.load(f_mask_in)
        tiny_mask = load_mask_by_name(mask_data, 'tiny_mask')
        confusing_mask = load_mask_by_name(mask_data, 'confusing_mask')
    return tiny_mask, confusing_mask


# Save in new formats
def save_pkl_imgs(out_path, limg, rimg, left_img_path, calib=-1000):
    lzcompress_rate = 9
    with open(out_path, "wb") as f_img_out:
        img_data = {
            'left_image': {'data': lzcompress(limg, lzcompress_rate), 'shape': limg.shape, 'dtype': limg.dtype},
            'right_image': {'data': lzcompress(rimg, lzcompress_rate), 'shape': rimg.shape, 'dtype': rimg.dtype},
            'left_path': left_img_path,
            'calib': calib,
            'default_shape': limg.shape,
        }
        pkl.dump(img_data, f_img_out)


def save_pkl_disp(out_path, disp, tiny_mask, confusing_mask):
    lzcompress_rate = 9
    error = np.zeros_like(disp)
    with open(out_path, "wb") as f_disp_out:
        disp_data = {
            'left_disparity': {'data': lzcompress(disp, lzcompress_rate), 'shape': disp.shape, 'dtype': disp.dtype},
            'left_disparity_err': {'data': lzcompress(error, lzcompress_rate), 'shape': error.shape, 'dtype': error.dtype},
            'tiny_mask': {'data': lzcompress(tiny_mask, lzcompress_rate), 'shape': tiny_mask.shape, 'dtype': tiny_mask.dtype},
            'confusing_mask': {'data': lzcompress(confusing_mask, lzcompress_rate), 'shape': confusing_mask.shape, 'dtype': confusing_mask.dtype},
            'default_shape': disp.shape,
        }
        pkl.dump(disp_data, f_disp_out)


def save_pkl_segs(out_path, seg):
    lzcompress_rate = 9
    with open(out_path, "wb") as f_seg_out:
        seg_data = {
            'segmentation': {'data': lzcompress(seg, lzcompress_rate), 'shape': seg.shape, 'dtype': seg.dtype},
            'default_shape': seg.dtype
        }
        pkl.dump(seg_data, f_seg_out)



def assemble_paths(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    root_folders = [folder.strip() for folder in lines[0].split(',')]
    assembled_paths_2d = []
    for line in lines[1:]:
        filenames = [filename.strip() for filename in line.split(',')]
        current_paths = [f"{root_folder}/{filename}" for root_folder,
                         filename in zip(root_folders, filenames)]
        assembled_paths_2d.append(current_paths)
    return assembled_paths_2d


def _process_one(files, original_root, new_root):
    assert len(files) == 3, f'Expecting 3 files, got {len(files)}'
    if not all(os.path.exists(f) for f in files):
        _logger.error(f'Not all files exist: {files}')
        return False

    try:
        rgb_file, disparity_file, seg_file = files
        mask_file = rgb_file.replace('/Images/', '/Masks/')
        tiny_mask, confusing_mask = load_pkl_masks(mask_file)
        limg, rimg, left_img_path = load_pkl_imgs(rgb_file)
        disparity = load_pkl_disp(disparity_file)
        segs = load_pkl_segs(seg_file)

        new_rgb_file = rgb_file.replace(original_root, new_root)
        new_disparity_file = disparity_file.replace(original_root, new_root)
        new_seg_file = seg_file.replace(original_root, new_root)

        Path(new_rgb_file).parent.mkdir(parents=True, exist_ok=True)
        Path(new_disparity_file).parent.mkdir(parents=True, exist_ok=True)
        Path(new_seg_file).parent.mkdir(parents=True, exist_ok=True)

        save_pkl_imgs(new_rgb_file, limg, rimg, left_img_path)
        save_pkl_disp(new_disparity_file, disparity, tiny_mask, confusing_mask)
        save_pkl_segs(new_seg_file, segs)
        _logger.info(f'Processed {rgb_file}')
    except Exception as e:
        _logger.error(f'Error processing {rgb_file}: {e}')
        return False

    new_root = Path(new_root)
    return (
        str(Path(new_rgb_file).relative_to(new_root)),
        str(Path(new_disparity_file).relative_to(new_root)),
        str(Path(new_seg_file).relative_to(new_root)),
    )


def _main(args):
    file_list_path = args.filelist
    assert os.path.exists(file_list_path), f'File not found: {file_list_path}'
    filepaths_2d = assemble_paths(file_list_path)
    _logger.info(f'Found {len(filepaths_2d)} groups of images.')

    original_root = str(Path(file_list_path).absolute().parent.parent.absolute())
    new_root = Path(args.output_dir)
    new_root.mkdir(parents=True, exist_ok=True)
    new_root = str(new_root)
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(_process_one, original_root=original_root, new_root=new_root), filepaths_2d)

    # generate filelist and check the number of successfully processed files
    num_valid = 0
    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = Path(args.output_dir)
    filelist_dir = output_dir / 'file_list'
    filelist_dir.mkdir(parents=True, exist_ok=True)
    filelist_path = filelist_dir / f'pkl_filelist_fisheye_{time_str}.csv'
    with open(filelist_path, 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow([str(output_dir.absolute())] * 3)

        for item in results:
            if item:
                num_valid += 1
                writer.writerow(item)

    print('Valid rate (%d/%d): %.2f' %
          (num_valid, len(results), num_valid/len(results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process CSV file with file paths.")
    parser.add_argument('-f', '--filelist',
                        help="Path to the CSV file containing file paths.")
    parser.add_argument('-o', '--output-dir',
                        help="Path to the output directory.")
    args = parser.parse_args()
    _main(args)
