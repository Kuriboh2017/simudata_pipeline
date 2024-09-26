import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.measure_convert import disp_to_depth
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb

def warp(rimg, ldisp):
    H, W, C = rimg.shape
    pad_size = 300
    rimg_pad = cv2.copyMakeBorder(rimg, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0,0,0)).astype(np.float32)

    hs = np.linspace(0, H-1, H).astype(np.int16)
    ws = np.linspace(0, W-1, W).astype(np.int16)

    wv, hv = np.meshgrid(ws, hs)
    wv_r = wv - ldisp
    wv_r_ceil = np.ceil(wv_r)
    wv_r_flor = np.floor(wv_r)

    dist_to_flor = wv_r - wv_r_flor
    dist_to_flor = np.repeat(np.expand_dims(dist_to_flor, 2), C, axis=2)

    rimg_pad_ceil = rimg_pad[hv+pad_size, wv_r_ceil.astype(np.int16)+pad_size, :]
    rimg_pad_flor = rimg_pad[hv+pad_size, wv_r_flor.astype(np.int16)+pad_size, :]

    limg_rec = dist_to_flor * rimg_pad_ceil + (1-dist_to_flor) * rimg_pad_flor
    limg_rec = limg_rec.astype(np.uint8)

    return limg_rec


def compare_fig_stitch(cfgs, show_lst):

    num_row = len(show_lst)
    num_col = len(show_lst[0])

    edge_pad = 10
    width_lst  = [edge_pad]
    height_lst = [edge_pad]

    for r in range(num_row):
        show_item = show_lst[r][0]
        h, w = show_item['data'].shape[:2]

        height_lst.append(h)
        height_lst.append(edge_pad)

    for c in range(num_col):
        show_item = show_lst[0][c]
        h, w = show_item['data'].shape[:2]

        width_lst.append(w)
        width_lst.append(edge_pad)
            
    
    w_total = sum(width_lst)
    h_total = sum(height_lst)
    canvas = np.ones((h_total, w_total, 3)) * 155

    w_head, w_tail = 0, 0
    h_head, h_tail = 0, 0

    for r in range(num_row):
        for c in range(num_col):
            show_item = show_lst[r][c]
            
            w_head = sum(width_lst[:2*c+1])
            w_tail = w_head + width_lst[2*c+1]

            h_head = sum(height_lst[:2*r+1])
            h_tail = h_head + height_lst[2*r+1]


            if show_item['type'] in ['img', 'img_cls'] :
                canvas[h_head:h_tail, w_head:w_tail, :] = show_item['data'].astype(np.uint8)
            
            if show_item['type'] == 'disp':
                depth_data = disp_to_depth(cfgs, show_item['data'])
                depth_data = 255 * depth_data / depth_data.max()
                depth_vis  = cv2.applyColorMap(depth_data.astype(np.uint8), cv2.COLORMAP_JET)
                canvas[h_head:h_tail, w_head:w_tail, :] = depth_vis
    

    return canvas



def compare_fig(cfgs, show_lst):

    num_row = len(show_lst)
    num_col = len(show_lst[0])

    H,  W = show_lst[0][0]['data'].shape[:2]
    input_size  = (W, H)

    width  = np.ceil(input_size[0] * num_col / 100) * 1.3
    height = np.ceil(input_size[1] / 100) * 2 * 1.2
    fig, axs_lst = plt.subplots(num_row, num_col, figsize=(width, height))  # , gridspec_kw={'width_ratios': [1, 1.06, 1.06, 1]}) , gridspec_kw={'height_ratios': [1, 2]}
    if num_row == 1: axs_lst = [axs_lst]
    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.90, wspace=0.15, hspace=0.15)


    disp_max = -1
    for show_data_row in show_lst:
        for show_data in show_data_row:
            if show_data['type'] == 'disp':
                if show_data['data'] is not None:
                    max_val = np.max(show_data['data'])
                    if max_val > disp_max:
                        disp_max = max_val

    disp_max = 60

    for row_idx, axs_row in enumerate(axs_lst):
        for col_idx, axs in enumerate(axs_row):

            show_data = show_lst[row_idx][col_idx]['data']
            show_type = show_lst[row_idx][col_idx]['type']

            if show_data is not None:

                if show_type == 'img':
                    # show_img = cv2.cvtColor(show_data, cv2.COLOR_BGR2RGB)
                    axs.imshow(show_data, aspect="auto")
                elif show_type == 'img_cls':
                    axs.imshow(show_data, aspect="auto")
                elif show_type == 'disp':
                    im = axs.imshow(disp_to_depth(cfgs, show_data), cmap=plt.get_cmap('jet'), aspect="auto", vmax=disp_max, vmin=0)
                    divider = make_axes_locatable(axs)
                    cax = divider.append_axes("right", size="1%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                elif show_type == 'disp_diff':
                    show_data_ref = show_lst[row_idx][col_idx]['data_ref']
                    im = axs.imshow(disp_to_depth(cfgs, show_data) - disp_to_depth(cfgs, show_data_ref), cmap=plt.get_cmap('jet'), aspect="auto", vmax=10, vmin=0)
                    divider = make_axes_locatable(axs)
                    cax = divider.append_axes("right", size="1%", pad=0.05)
                    fig.colorbar(im, cax=cax)

    fig.canvas.draw()
    fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return fig_data



def histogram(model_name, data_lst, num_bin=100):
    data_arr = np.array(data_lst)
    plt.hist(data_arr, bins=num_bin, range=[0,4])
    plt.savefig('hist/hist_%s.png' % model_name)
    plt.clf()


def bar_fig(cfgs, model_name, data_lst):
    x_tick_start = cfgs.hist.range[0]
    x_tick_end   = cfgs.hist.range[1]
    x_coords = np.linspace(x_tick_start, x_tick_end, cfgs.hist.num_bin)
    
    plt.bar(x_coords, data_lst)
    plt.savefig('hist/bar_%s.png' % model_name)
    plt.clf()