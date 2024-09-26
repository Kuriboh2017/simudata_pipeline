#!/usr/bin/env python3
from lz4.frame import decompress as lzdecompress
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pickle as pkl


def assemble_paths(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    root_folder = [folder.strip() for folder in lines[0].split(',')][0]
    assembled_paths_2d = []
    for line in lines[1:]:
        filenames = [filename.strip() for filename in line.split(',')]
        current_paths = [f"{root_folder}/{f}" for f in filenames]
        relative_paths = [str(Path(root_folder) / Path(f))
                          for f in current_paths]
        assembled_paths_2d.append(relative_paths)
    return assembled_paths_2d


def load_pkl_imgs(img_path):
    with open(img_path, "rb") as f_img_in:
        img_data = pkl.load(f_img_in)
        limg = np.frombuffer(lzdecompress(
            img_data['left_image']['data']), dtype=img_data['left_image']['dtype'])
        limg = limg.reshape(img_data['left_image']['shape'])
    return limg


def load_pkl_disp(disp_path):
    with open(disp_path, "rb") as f_disp_in:
        disp_data = pkl.load(f_disp_in)
        disp = np.frombuffer(lzdecompress(
            disp_data['left_disparity']['data']), dtype=disp_data['left_disparity']['dtype'])
        disp = disp.reshape(disp_data['left_disparity']['shape'])
    return disp


def load_pkl_segs(seg_path):
    with open(seg_path, "rb") as f_seg_in:
        seg_data = pkl.load(f_seg_in)
        seg = np.frombuffer(lzdecompress(
            seg_data['segmentation']['data']), dtype=seg_data['segmentation']['dtype'])
        seg = seg.reshape(seg_data['segmentation']['shape'])
    return seg


def check_data_shape(path_group):
    img, disp, seg = path_group
    img_data = load_pkl_imgs(img)
    disp_data = load_pkl_disp(disp)
    seg_data = load_pkl_segs(seg)
    if img_data.shape != (704, 1280, 3):
        print(f'Error: img_data.shape = {img_data.shape}, path = {img}')
    if disp_data.shape != (704, 1280):
        print(f'Error: disp_data.shape = {disp_data.shape}, path = {disp}')
    if seg_data.shape != (704, 1280):
        print(f'Error: seg_data.shape = {seg_data.shape}, path = {seg}')


def check_filelist(filepath):
    paths = assemble_paths(filepath)
    # # Sequential checking:
    # for path in paths:
    #     check_data_shape(path)
    # Parallel checking:
    with Pool(cpu_count()) as pool:
        pool.map(check_data_shape, paths)


check_filelist(
    '/mnt/112-data/R23024/data/junwu/data/filelist_erp/fy_synt_0918_val_lz4y.csv')

check_filelist(
    '/mnt/112-data/R23024/data/junwu/data/filelist_erp/fy_synt_0918_train_lz4y.csv')
