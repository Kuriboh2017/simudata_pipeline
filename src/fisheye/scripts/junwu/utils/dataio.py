import os 
import re
import cv2
import webp
import joblib
import numpy as np
import pickle as pkl
from tqdm import tqdm

from utils.viz import compare_fig, compare_fig_stitch
from lz4.frame import compress as lzcompress
from lz4.frame import decompress as lzdecompress

import pdb


def pklload(filename):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    return data

def decomress(data, dtype):
    data = lzdecompress(data)
    data = np.frombuffer(data, dtype=dtype)

    return data


def load_pkl_yby2(data_path, k_list=['data_one','data_two'], k_info_list=['shape', 'dtype'] ):
    data = pklload(data_path)
    return_data = {}

    data_shape = data[k_info_list[0]]
    data_dtype = data[k_info_list[1]]

    for k, v in data.items():
        if k in k_list:
            if v is not None:
                v = decomress(v, data_dtype)
                v = v.reshape(data_shape)
                return_data[k] = v
            else:
                return_data[k] = np.zeros(data_shape)
    if len(k_info_list) > 2:
        return [return_data[k] for k in k_list] + [data[k_info_list[i]] for i in range(2, len(k_info_list))]
    else:
        return [return_data[k] for k in k_list]


def load_npz_seg(datas_path, dataw_path):
    segs = np.load(datas_path)['arr']
    segw = np.load(dataw_path)['arr']

    combined_seg = np.zeros_like(segs, dtype=np.uint8) #combined_seg[left_seg == 1] = 0
    combined_seg[segs == 0] = 1
    combined_seg[segw == 1] = 2  # 0:sky, 1:other, 2:wire.
    combined_seg[segs == 255] = 255
    combined_seg[segw == 255] = 255 # 255 ignore
    
    return combined_seg


def load_npz_disp(data_path):
    data = np.load(data_path)
    disp = data['disp_left']
    derr = data['error_left']
    return disp, derr

    # if '3to1' in data_path:
    #     return np.load(data_path)['arr_0'].astype('float16')
    # else:
    #     return np.load(data_path)['data'].astype('float16')

def load_pkl_img(data_path):
    l_img, r_img = load_pkl_yby2(data_path, k_list=['left_image', 'right_image'], k_info_list=['image_shape', 'image_dtype'])
    
    l_img = l_img.astype(np.uint8)
    r_img = r_img.astype(np.uint8)

    return l_img, r_img

def load_pkl_img_with_path(data_path):
    l_img, r_img, lpath = load_pkl_yby2(data_path, k_list=['left_image', 'right_image'], k_info_list=['image_shape', 'image_dtype', 'left_path'])
    
    l_img = l_img.astype(np.uint8)
    r_img = r_img.astype(np.uint8)

    return l_img, r_img, lpath


def load_pkl_disp(data_path):
    disp, error = load_pkl_yby2(data_path, k_list=['left_disparity', 'left_disparity_err'], k_info_list=['disparity_shape', 'disparity_dtype'])
    return disp.copy(), error.copy()

def load_pkl_disp_with_selmap(data_path):
    disp, error, sel_map, map_id = load_pkl_yby2(data_path, k_list=['left_disparity', 'left_disparity_err'], k_info_list=['disparity_shape', 'disparity_dtype', 'selection_map', 'map_id'])
    return disp.copy(), error.copy(), sel_map.copy(), map_id.copy()

def load_pkl_seg(data_path):
    seg = load_pkl_yby2(data_path, k_list=['segmentation'], k_info_list=['segment_shape', 'segment_dtype'])
    return seg[0]


# def load_webp_img(left_img_path):
#     right_img_path = left_img_path.replace('cam1_0', 'cam1_1')
#     l_img  = np.asarray(webp.load_image(left_img_path, 'RGB'))
#     r_img  = np.asarray(webp.load_image(right_img_path, 'RGB'))

#     return l_img, r_img


def load_webp_img(left_img_path):
    right_img_path = left_img_path.replace('cam0_0', 'cam0_1').replace('cam1_0', 'cam1_1')
    l_img  = cv2.imread(left_img_path)
    r_img  = cv2.imread(right_img_path)

    return l_img, r_img

def load_png_img(left_img_path, right_img_path):
    l_img  = cv2.imread(left_img_path)
    r_img  = cv2.imread(right_img_path)
    return l_img, r_img


def load_webp_seg(seg_path):
    seg  = np.asarray(webp.load_image(seg_path, 'RGB'))
    return seg


def load_joblib_lite_disp(disp_path):
    org_shape = (681, 1280)
    data = joblib.load(disp_path)

    disp  = data["disp"].astype(np.float32)
    error = data["error"].astype(np.float32)

    disp_h, disp_w = disp.shape[-2:]
    r = org_shape[-1] / disp.shape[-1]
    disp = disp.transpose(1, 2, 0)
    h_tmp = int(disp_h * r)
    w_tmp = int(disp_w * r)
    disp = cv2.resize(disp, (w_tmp, h_tmp), interpolation=cv2.INTER_LINEAR)
    disp = disp[:org_shape[0]] * r
    # MUCHUN write in 20230408
    error = error.transpose(1,2,0)
    error = cv2.resize(error, org_shape[::-1], interpolation=cv2.INTER_LINEAR)
    
    return disp[..., 0].copy(), error[..., 0].copy()


def load_opencv_disp(disp_path, H=1362, W=1280):

    disp = np.load(disp_path, allow_pickle=True)['L2R_disparity']
    disp = disp.reshape(H, W)
    disp = disp[0::2].astype(np.float16)/16

    return disp


def load_opencv_disp2(disp_path, H=681, W=1280):

    disp = np.load(disp_path, allow_pickle=True)['disp_left']
    disp = disp.reshape(H, W)
    disp = disp.astype(np.float16)/16

    derr = np.load(disp_path, allow_pickle=True)['error_left']

    return disp, derr


def export_imgs_for_review(limg_path, limg_data, rimg_data, disp_data, seg_data, disp_pred=None):
    file_id = limg_path.split('/')[-1].split('.')[0]

    disp_data = disp_data * 255 / disp_data.max()
    seg_data = seg_data.copy()
    seg_data[seg_data==2] = 255
    seg_data[seg_data==1] = 126

    os.makedirs('review', exist_ok=True)
    cv2.imwrite('review/limg_%s.png' % file_id, limg_data)
    cv2.imwrite('review/rimg_%s.png' % file_id, rimg_data)
    cv2.imwrite('review/disp_%s.png' % file_id, disp_data.astype(np.uint8))
    cv2.imwrite('review/segs_%s.png' % file_id, cv2.applyColorMap(seg_data.astype(np.uint8), cv2.COLORMAP_INFERNO))

    if disp_pred is not None:
        disp_pred = disp_pred * 255 / disp_pred.max()
        cv2.imwrite('review/disp_pred_%s.png' % file_id, disp_pred.astype(np.uint8))


def export_compare_fig(cfgs, limg_path, subpath, show_lst):
    map_name = limg_path.split('/')[11]
    file_id = limg_path.split('/')[-1].split('.')[0]
    # fig_data = compare_fig(cfgs, show_lst)
    fig_data = compare_fig_stitch(cfgs, show_lst)

    # return fig_data
    os.makedirs('review/%s/' % subpath, exist_ok=True)
    cv2.imwrite('review/%s/compare_%s_%s.png' % (subpath, map_name, file_id), fig_data.astype(np.uint8))


def get_synt_token(path_split):
    # kwd_pattern  = 'Autonomy|\d{8}_out|\d{8}_high_out|sensors_data|group|cam'
    kwd_pattern = 'group'
    
    group_idx = -1
    for item_idx, item in enumerate(path_split):
        if re.match(kwd_pattern, item):
            group_idx = item_idx
            break 
    matched_item   = [path_split[group_idx-1], path_split[-1].split('.')[0]]  # path_split[group_idx-2], 
    matched_folder = [path_split[group_idx-2], path_split[group_idx-1]]

    return '_'.join(matched_item) # , '_'.join(matched_folder)


def get_real_token(path_split):
    
    auto_idx = -1
    return_lst = []
    for item_idx, item in enumerate(path_split):
        if 'Autonomy' in item:
            return_lst.append(item)
            return_lst.append(path_split[item_idx+1])

        if 'common-pipeline-data' in item:
            return_lst.append(item)

        if 'group' in item:
            return_lst.append(item)

        if 'cam' in item:
            return_lst.append(item)
        
        if '.pkl' in item:
            return_lst.append(item)

    return '_'.join(return_lst)


def parse_token_dict(data_root_lst):
    return_dict = {}
    num_sample = 0
    for data_root in data_root_lst:
        for root, dirs, files in tqdm(os.walk(data_root, topdown=True)):
            for name in files:
                file_path = os.path.join(root, name)

                if ('.webp' in file_path) and ('cam1_0' in file_path or 'cam0_0' in file_path) and ('Image' in file_path):
                    path_split = file_path.split('/')
                    token = get_synt_token(path_split)
                    return_dict[token] = file_path
                    num_sample += 1

    return return_dict, num_sample


def parse_file_list(data_root_lst):
    return_list = []
    for data_root in data_root_lst:
        for root, dirs, files in tqdm(os.walk(data_root, topdown=True)):
            for name in files:
                file_path = os.path.join(root, name)

                if ('.webp' in file_path) and ('cam1_0' in file_path or 'cam0_0' in file_path) and ('Image' in file_path):
                    return_list.append(file_path)

    return return_list

def check_exists(path_lst):
    flag = True
    for path in path_lst:
        flag = flag and os.path.exists(path)
        # print(os.path.exists(path), path)
    return flag

def parse_imgdisp_file_list(filelst, seg_root, opcv_root):
    """
    parse the image path, disparity path segmentation path (infer from segroot) of each sample from filslst
    """
    return_dict = []
    num_sample = 0
    num_exists = 0
    imgs_root = ''
    disp_root = ''

    with open(filelst, 'r') as fin:
        for line_idx, line in tqdm(enumerate(fin)):
            line_split = line.strip().split(',')

            # if line_idx > 400: break
            if line_idx == 0:
                imgs_root = line_split[0]
                disp_root = line_split[2]
            else:
                sub_limg_path = line_split[0]
                sub_rimg_path = line_split[1]
                sub_disp_path = line_split[2]

                full_limg_path = os.path.join(imgs_root, sub_limg_path)
                full_rimg_path = os.path.join(imgs_root, sub_rimg_path)
                full_disp_path = os.path.join(disp_root, sub_disp_path)
                full_segs_path = full_limg_path.replace(imgs_root, seg_root).replace('/Image/', '/Segment_sky_647d31/').replace('.png', '.npz')  # seg path for sky
                full_segw_path = full_limg_path.replace(imgs_root, seg_root).replace('/Image/', '/Segment_wire_5dbdc5/').replace('.png', '.npz') # seg path for wire
                full_opcv_path = full_limg_path.replace(imgs_root, opcv_root).replace('/Image/', '/Image/result/').replace('.png', '.npz')
                
                path_exists = check_exists([full_limg_path, full_rimg_path, full_disp_path, full_segs_path, full_segw_path, full_opcv_path])
                # pdb.set_trace()
                if path_exists:
                    return_dict.append(
                        {
                            'limg_path': full_limg_path,
                            'rimg_path': full_rimg_path,
                            'disp_path': full_disp_path,
                            'segs_path': full_segs_path,
                            'segw_path': full_segw_path,
                            'opcv_path': full_opcv_path
                        }
                    )
                    
                    num_exists += 1
                num_sample += 1

    return return_dict, num_exists, num_sample


def parse_all_fromfilelst(filelst, with_token=False):
    """
    parse the image path, disparity path and segmentation path of each sample from filslst
    """
    return_dict = {} if with_token else []

    num_sample = 0

    data_root = ''
    with open(filelst, 'r') as fin:
        for line_idx, line in tqdm(enumerate(fin)):
            line_split = line.strip().split(',')
            if line_idx == 0:
                data_root = line_split[0]
            else:
                sub_imgs_path = line_split[0]
                sub_disp_path = line_split[1]
                sub_segs_path = line_split[2]

                full_imgs_path = os.path.join(data_root, sub_imgs_path)
                full_disp_path = os.path.join(data_root, sub_disp_path)
                full_segs_path = os.path.join(data_root, sub_segs_path)

                if with_token:
                    token = get_real_token(sub_imgs_path.split('/'))
                    return_dict[token] = {
                        'imgs_path': full_imgs_path,
                        'disp_path': full_disp_path,
                        'segs_path': full_segs_path
                    }
                else:
                    return_dict.append(
                        {
                            'imgs_path': full_imgs_path,
                            'disp_path': full_disp_path,
                            'segs_path': full_segs_path
                        }
                    )
                
                num_sample += 1

    return return_dict, num_sample


    

    
def save_pkl_imgs(out_path, limg, rimg, left_img_path):
    # compress and save data
    lzcompress_rate = 9
    with open(out_path, "wb") as f_img_out:
        img_data = {
            'left_image'  : lzcompress(limg, lzcompress_rate),
            'right_image' : lzcompress(rimg, lzcompress_rate),
            'image_shape' : limg.shape,
            'image_dtype' : limg.dtype,
            'left_path'   : left_img_path
        }
        pkl.dump(img_data, f_img_out)

def save_pkl_segs(out_path, seg):
    # compress and save data
    lzcompress_rate = 9
    with open(out_path, "wb") as f_seg_out:
        seg_data = {
            'segmentation'  : lzcompress(seg, lzcompress_rate),
            'segment_shape' : seg.shape,
            'segment_dtype' : seg.dtype
        }
        pkl.dump(seg_data, f_seg_out)


def save_pkl_disp(out_path, disp, derr, sel_idx_map):
    # compress and save data
    lzcompress_rate = 9
    with open(out_path, "wb") as f_disp_out:
        disp_data = {
            'left_disparity'     : lzcompress(disp, lzcompress_rate),
            'left_disparity_err' : lzcompress(derr, lzcompress_rate),
            'selection_map'      : sel_idx_map,
            'map_id'             : {'large':0, 'lite':1, 'opencv':2, '3drecon':3},
            'disparity_shape'    : disp.shape,
            'disparity_dtype'    : disp.dtype
        }
        pkl.dump(disp_data, f_disp_out)