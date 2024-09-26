import os
import re
import cv2
import csv
import hydra
import imageio
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path 
import multiprocessing.pool
from omegaconf import DictConfig

from utils.viz import warp, compare_fig_stitch
from utils.dataio import parse_token_dict, parse_file_list
from utils.dataio import load_npz_seg, load_npz_disp, load_webp_img
from utils.dataio import save_pkl_imgs, save_pkl_disp, save_pkl_segs

import pdb


def output_paths(cfgs, in_limg_path):

    kwd_pattern = 'group'
    path_split  = in_limg_path.split('/')

    group_idx = -1
    for item_idx, item in enumerate(path_split):
        if re.match(kwd_pattern, item):
            group_idx = item_idx
            break
    
    out_imgs_folder = os.path.join( *([cfgs.output_root] + path_split[group_idx-1: group_idx+3]))
    out_disp_folder = out_imgs_folder.replace('Image', 'Disparity')
    out_segs_folder = out_imgs_folder.replace('Image', 'Segment')

    os.makedirs(out_imgs_folder, exist_ok=True)
    os.makedirs(out_disp_folder, exist_ok=True)
    os.makedirs(out_segs_folder, exist_ok=True)

    out_imgs_path = os.path.join(out_imgs_folder, path_split[-1]).replace('.webp', '.pkl')
    out_disp_path = os.path.join(out_disp_folder, path_split[-1]).replace('.webp', '.pkl')
    out_segs_path = os.path.join(out_segs_folder, path_split[-1]).replace('.webp', '.pkl')

    return out_imgs_path, out_disp_path, out_segs_path


def process_one(info):
    cfgs = info[0]
    src_limg_path = info[1]

    src_disp_path = src_limg_path.replace('Image', 'Disparity').replace('.webp', '.npz')
    src_segs_path = src_limg_path.replace('Image', 'Segmentation/Graymap').replace('.webp', '.npz')
    
    try:
        # load fishlized data
        limg, rimg = load_webp_img(src_limg_path)
        disp = load_npz_disp(src_disp_path)
        segs = load_npz_seg(src_segs_path)
        

        # save loaded data as pkl files
        out_imgs, out_disp, out_segs = output_paths(cfgs, src_limg_path)
        save_pkl_imgs(out_imgs, limg, rimg, src_limg_path)
        save_pkl_disp(out_disp, disp)
        save_pkl_segs(out_segs, segs)
    except:
        return False
    
    return (
        Path(out_imgs).relative_to(cfgs.output_root), 
        Path(out_disp).relative_to(cfgs.output_root), 
        Path(out_segs).relative_to(cfgs.output_root)
    )


def review(cfgs, path_lst):

    for item in tqdm(path_lst[::1000]):
        limg_path = item
        rimg_path = item.replace('cam0_0', 'cam0_1').replace('cam1_0', 'cam1_1')
        disp_path = item.replace('Image', 'Disparity').replace('.webp', '.npz')
        segs_path = item.replace('Image', 'Segmentation/Graymap').replace('.webp', '.npz')

        limg = cv2.imread(limg_path)
        rimg = cv2.imread(rimg_path)
        disp = np.load(disp_path)['arr_0']
        segs = np.load(segs_path)['arr_0']

        show_lst_1st = [[]]
        show_lst_1st[0].append({'data': limg, 'type':'img'})
        show_lst_1st[0].append({'data': limg, 'type':'img'})
        show_lst_1st[0].append({'data': disp, 'type':'disp'})
        show_lst_1st[0].append({'data': cv2.applyColorMap((segs*127.5).astype(np.uint8), cv2.COLORMAP_INFERNO), 'type':'img_cls'})
        fig_1st = compare_fig_stitch(cfgs, show_lst_1st)

        show_lst_2nd = [[]]
        show_lst_2nd[0].append({'data': rimg, 'type':'img'})
        show_lst_2nd[0].append({'data': warp(rimg, disp), 'type':'img'})
        show_lst_2nd[0].append({'data': disp, 'type':'disp'})
        show_lst_2nd[0].append({'data': cv2.applyColorMap((segs*127.5).astype(np.uint8), cv2.COLORMAP_INFERNO), 'type':'img_cls'})
        fig_2nd = compare_fig_stitch(cfgs, show_lst_2nd)

        gif_fig = [fig_1st, fig_2nd]

        os.makedirs('review/top', exist_ok=True)
        os.makedirs('review/bot', exist_ok=True)

        file_id = limg_path.split('/')[-1].split('.')[0]
        if 'cam1' in limg_path:
            imageio.mimsave('review/top/compare_%s.gif' % file_id, gif_fig, duration=0.6)
        else:
            imageio.mimsave('review/bot/compare_%s.gif' % file_id, gif_fig, duration=0.6)




@hydra.main(version_base=None, config_path='config', config_name='synt.yaml')
def main(cfgs: DictConfig):

    src_limg_list = parse_file_list(cfgs.root_path_lst)


    # review files
    # review(cfgs, src_limg_list)
    # return 


    # process files
    args_lst = [(cfgs, src_limg_path) for src_limg_path in src_limg_list ]
    num_thread = 80
    with multiprocessing.pool.ThreadPool(processes=num_thread) as pool:
        return_list = list(tqdm(pool.imap(process_one, args_lst), total=len(args_lst)))



    # generate filelist and check the number of successfully processed files
    num_valid = 0
    with open('pkl_filelist_fisheye_lize_0821.csv', 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow([cfgs.output_root] * 3)

        for item in return_list:
            if item:
                num_valid += 1
                writer.writerow(item)

    print('valid rate (%d/%d): %.2f' % (num_valid, len(return_list), num_valid/len(return_list)))

        

if __name__ == '__main__':
    main()