import os
import cv2
import numpy as np
# from multiprocessing import Pool, cpu_count
# import multiprocessing
# from subprocess import run
# from functools import partial

train_list_3= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/all/pkl_filelist_rect_fine_train_all_lz4y.csv'
test_list_3=  '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/all/pkl_filelist_rect_fine_val_all_lz4y.csv' # real val
synt_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/fy_synt_0918_train_lz4y.csv' #12W  14W(202402181035) 72W(20240428)
synt_test_list=  '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/fy_synt_0918_val_lz4y.csv' # 3W 4W(202402181035) 20W(20240428)
near_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/near/pkl_filelist_rect_fine_train_near_lz4y.csv' #18W
near_test_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/near/pkl_filelist_rect_fine_val_near_lz4y.csv' #6196
# powerline substation real+synt
substation_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/substation/pkl_filelist_rect_fine_train_substation_lz4y.csv' #18W 13W(20240428)
substation_test_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/substation/pkl_filelist_rect_fine_val_substation_lz4y.csv'
# tiny objects
tiny_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/tiny_train_lz4y.csv' #25W 37W(202402181035)
tiny_test_list= '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/tiny_val_lz4y.csv' #7315(202402181035)

# indoor
indoor_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/indoor/pkl_filelist_rect_fine_train_all_indoor_lz4y.csv' #20240412 9.5W 11W(109828,20240428)
indoor_test_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/indoor/pkl_filelist_rect_fine_val_all_indoor_lz4y.csv'

# debug:
# powerhouse real+synt
powerhouse_train_list= '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/powerhouse/pkl_filelist_rect_fine_train_powerhouse_lz4y.csv'
powerhouse_test_list= '/mnt/119-data/R22612/Data/lite_filelist/powerhouse/pkl_filelist_rect_fine_val_powerhouse_lz4y.csv' #only real data



badcase_far_rgb= '/mnt/119-data/R22612/Data/ERP/filelist/badcase_filelist/farbad_water_0918.csv'    # include far and water 26W
badcase_far_water= '/mnt/119-data/R22612/Data/ERP/filelist/badcase_filelist/fog_glare_0918.csv'       # fog and glare 12.5W
badcase_far_wire= '/mnt/119-data/R22612/Data/ERP/filelist/badcase_filelist/far_wire_0918.csv'   # 5k


from dataset import load_pkl_imgs_0918

def load_img_write(pkl_path): #with write remove
    path_key = pkl_path.split('/')    
    # left image folder pattern: /cam0_0/, /cam1_0/, /left/
    cam_list = []   
    left_list = []    
    cam_left_list = []
    for i in range(len(path_key)):
        if path_key[i].startswith('cam') and path_key[i].endswith('_0') and  ((i+1)<len(path_key) and path_key[i+1] == 'left'):
            cam_left_list.append(i)
    if len(cam_left_list) !=1:
        for i in range(len(path_key)):
            if path_key[i].startswith('cam') and path_key[i].endswith('_0'):
                cam_list.append(i)
            if path_key[i] == 'left':
                left_list.append(i)

    if (len(cam_list)  + len(left_list) + len(cam_left_list) ) !=1:
        print('Error. cam_list',[path_key[i] for i in cam_list],[path_key[i] for i in left_list],[path_key[i] for i in cam_left_list],pkl_path)
        return None
    else:
        # print('reading',pkl_path)
        if len(cam_list) == 1:
            path_key[cam_list[0]] = path_key[cam_list[0]][:-1]+'1'
            img_path_right = '/'.join(path_key)
        elif len(left_list) == 1:
            path_key[left_list[0]] = 'right'
            img_path_right = '/'.join(path_key)
        elif len(cam_left_list) == 1:
            path_key[cam_left_list[0]+1] = 'right'
            img_path_right = '/'.join(path_key)            
        img_path_left = os.path.splitext(pkl_path)[0] + '.webp'
        img_path_right = os.path.splitext(img_path_right)[0] + '.webp'
            
        # print('pkl_path',pkl_path,'img_path_left',img_path_left,'img_path_right',img_path_right)
        if os.path.exists(pkl_path):
            img_l,img_r,lpath,calib = load_pkl_imgs_0918(pkl_path)
            # print('img_l',img_l.shape,'img_r',img_r.shape,'lpath',lpath,'calib',calib)
            if ('/mnt/103-data1/R22612/Data/' in pkl_path or '/mnt/103-data/R22612/Data/' in pkl_path):
                # print('Read img_path_left',img_path_left,'img_path_right',img_path_right)
                cv2.imwrite(img_path_left,img_l)
                os.makedirs(os.path.dirname(img_path_right), exist_ok=True)
                cv2.imwrite(img_path_right,img_r)
                txt_p = os.path.splitext(pkl_path)[0] + '.txt'
                with open(txt_p,'w') as f:
                    f.write('left_path,'+lpath+'\n')
                    f.write('calib,'+str(calib)+'\n')
                if os.path.exists(img_path_left) and os.path.exists(img_path_right) and os.path.exists(txt_p):
                    img_l_tmp = cv2.imread(img_path_left)
                    img_r_tmp = cv2.imread(img_path_right)
                    if img_l_tmp is not None and img_r_tmp is not None:
                        os.remove(pkl_path)
                        print('Removed pkl_path',pkl_path)
                        return 1
                    else:
                        print(img_l_tmp is None , img_r_tmp is None)
                        return 0
        else:
            # img_l = cv2.imread(img_path_left)
            # img_r = cv2.imread(img_path_right)
            # print('Read img_path_left',img_path_left,'img_path_right',img_path_right)
            # print('webp',pkl_path)
            return 0
            pass
        return 0
        # return img_l, img_r,'lpath',-1000   
    print('Error. pkl_path',pkl_path)

csv_list=[train_list_3,test_list_3,synt_train_list,synt_test_list,near_train_list,near_test_list,substation_train_list,substation_test_list,tiny_train_list,tiny_test_list,indoor_train_list,indoor_test_list,powerhouse_train_list,powerhouse_test_list]
csv_far_list=[badcase_far_rgb,badcase_far_water,badcase_far_wire]

# csv_list=[near_test_list]
# csv_far_list=[badcase_far_rgb,badcase_far_water,badcase_far_wire]

def replace_root_103(
                        data_roots,
                        old_dir = '/mnt/119-data/R22612/Data/ERP', # '/mnt/112-data/data/junwu/data/',
                        new_dir = '/mnt/103-data1/R22612/Data/ERP',
                        ):        
    # old_dir = '/mnt/112-data/data/junwu/data/'
    # new_dir = '/mnt/114-data/R22612/Data/modelx_hires/train_high_y_lz4/'
    # print('Before 103 server old_dir',data_roots,[os.path.exists(x) for x in data_roots])
    data_roots = [x.replace(old_dir,new_dir) for x in data_roots]
    # print('After  103 server new_dir',data_roots,[os.path.exists(x) for x in data_roots])
    return data_roots

def get_p(list_input):
    #####  data list #####
    lrimg_lst = []
    ldisp_lst = []
    lrseg_lst = []
    with open(list_input) as f_lst:
        for line_idx, line in enumerate(f_lst):
            if line_idx == 0:
                data_roots = line.strip().split(',')
                data_roots = replace_root_103(data_roots)      
            else:
                data_paths = line.strip().split(',')
                if len(data_paths)!=3:
                    print('get_rgb_disp_seg_dict',line_idx,data_paths,list_input)
                    continue
                lrimg_lst.append(os.path.join(data_roots[0], data_paths[0]))
                ldisp_lst.append(os.path.join(data_roots[1], data_paths[1]))
                lrseg_lst.append(os.path.join(data_roots[2], data_paths[2]))
    return lrimg_lst, ldisp_lst, lrseg_lst

def get_p_far(list_input):
    #####  data list #####
    with open(list_input) as f:
        lines = [line.rstrip().split(',') for line in f.readlines()]
    left_far_rgb_list  = [x[0] for x in lines]
    right_far_rgb_list = [x[1] for x in lines]
    return left_far_rgb_list, right_far_rgb_list

def read_and_write(csv_list,far=False):    
    pkl_count =0
    for i in range(len(csv_list)):
        print('Start csv_list',csv_list[i])
        if far:
            lrimg_lst, ldisp_lst = get_p_far(csv_list[i])
        else:
            lrimg_lst, ldisp_lst, lrseg_lst = get_p(csv_list[i])
        s = np.zeros(len(lrimg_lst)).reshape(-1)
        for j in range(len(lrimg_lst)):
            if os.path.exists(lrimg_lst[j]):
                s[j] = 1
                # print('write',lrimg_lst[j])
                try:
                    load_img_write(lrimg_lst[j])
                except Exception as e: #except Watchdog:
                    print('Error',lrimg_lst[j],e)
                    
                # exit(0)
        ss = s.sum()
        pkl_count += ss
        print('Finish csv_list',csv_list[i],ss,ss/len(s),pkl_count)

if __name__ == '__main__':
    
    read_and_write(csv_list)
    #read_and_write(csv_far_list,far=True)