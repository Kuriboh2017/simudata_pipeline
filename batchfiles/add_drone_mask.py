from pathlib import Path
import argparse
import cv2
import lz4.frame as lz
import numpy as np
import os
import pickle as pkl
from scipy import interpolate

from resize_min_max_bilinear_class import Resize
from functools import partial
from multiprocessing import Pool, cpu_count
import glob
import shutil
resize = Resize()
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import random
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)


# 将numba的日志等级提高到warning，从而屏蔽njit的超长log
logging.getLogger('numba').setLevel(logging.WARNING)

MSK_PATH = '/mnt/119-data/samba-share/simulation/train/mx128_msk'
msk_files = [str(p) for p in Path(MSK_PATH).rglob('*.pkl') if p.is_file()]

def generate_mask_pkl(d):
    '''d:原始数据集路径'''
    if not os.path.exists(d):
        print(f'no path exists:{d}')
        return
    
    # 创建新目录
    ldir = d+r'_msk\left'
    rdir = d+r'_msk\right'
    os.makedirs(ldir,exist_ok=True)        
    os.makedirs(rdir,exist_ok=True)    
    # 读取视差图
    left_disp = [str(p) for p in Path(d+r'\cube_front\CubeDisparity_0.105').rglob('*.lz4') if p.is_file()]
    right_disp = [str(p) for p in Path(d+r'\cube_below\CubeDisparity_0.105').rglob('*.lz4') if p.is_file()]
    with Pool(cpu_count()-10) as p:
        p.map(partial(generate_one_mask_pkl, save_dir=ldir), left_disp)
    with Pool(cpu_count()-10) as p:
        p.map(partial(generate_one_mask_pkl, save_dir=rdir), right_disp)    

def generate_one_mask_pkl(disp_dir, save_dir):
    '''用视差图查找对应分割、RGB图'''
    disp = _read_lz4(disp_dir)
    rgb = cv2.imread(disp_dir.replace('CubeDisparity_0.105', 'CubeScene').replace('lz4','webp'))
    seg = _read_lz4(disp_dir.replace('CubeDisparity_0.105',r'CubeSegmentation\Graymap'))
    msk = seg == 1
    # 打包pkl
    compression_level=3
    data={
        'disp':{'data':lz.compress(np.ascontiguousarray(disp),compression_level), 'type':disp.dtype,'shape':disp.shape},
        'mask':{'data':lz.compress(np.ascontiguousarray(msk),compression_level), 'type':msk.dtype, 'shape':msk.shape},
        'seg':{'data':lz.compress(np.ascontiguousarray(seg),compression_level), 'type':seg.dtype, 'shape':seg.shape},
        'rgb':{'data':lz.compress(np.ascontiguousarray(rgb),compression_level), 'type':rgb.dtype, 'shape':rgb.shape},
        'default_shape': disp.shape,
        'basline':0.105        
    }
    # 保存pkl
    file_name = os.path.basename(disp_dir).replace('lz4','pkl')
    save_name = os.path.join(save_dir, file_name)
    with open(save_name, 'wb') as msk_out:
        pkl.dump(data, msk_out)
    
    print(f'finished save{save_name}')
    
def read_msk_pkl(msk_dir, buff=''):
    if buff not in ['disp', 'mask', 'seg', 'rgb']:
        print('wrong buff name')
        return None
    if os.path.exists(msk_dir):
        with open(msk_dir, 'rb') as f:
            cdata = pkl.load(f)
        data = cdata[buff]
        arr = lz.decompress(data['data'])
        arr = np.frombuffer(arr, dtype = data['type'])
        arr = np.reshape(arr, data['shape'])
        return arr        
    
def _read_lz4(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    arr = lz.decompress(data['arr'])
    arr = np.frombuffer(arr, dtype=data['dtype'])
    arr = np.reshape(arr, data['shape'])
    return arr

def load_pkl(pkl_path):
    '''
    pkl_path:pkl文件路径名
    '''
    pkl_path = str(pkl_path)
    if not os.path.exists(pkl_path):
        print(f'pkl path not exist:{pkl_path}')
        return None
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_pkl(pkl_file, pkl_path):
    with open(pkl_path, 'wb') as f:
        pkl.dump(pkl_file, f)

def decompress_pkl(pkl_data):
    arr = lz.decompress(pkl_data['data'])
    arr = np.frombuffer(arr, dtype=pkl_data['dtype'])
    arr = np.reshape(arr, pkl_data['shape'])  
    return arr  

def show_pkl(pkl_path,key):
    pkl_data = load_pkl(pkl_path)
    pkl_data = decompress_pkl(pkl_data[key])
    if pkl_data is not None:
        plt.imshow(pkl_data, cmap='jet')
        plt.show()        
    
def dilation_msk(msk, d = 2):
    #expand = np.zeros_like(dpdata, dtype = np.float32)
    expand = np.zeros_like(msk, dtype = np.uint8)
    expand[msk] = 1
    # 膨胀特征
    d=1 + 2*d
    kernel = np.ones((d,d), np.uint8)
    # 实际操作是反向腐蚀，因为要让近处的深度覆盖远处的，即值小的优先
    #expand = cv2.dilate(expand, kernel, 1)
    expand = cv2.erode(expand, kernel, 1)   
        
    return expand==1

def combine_disp_pkl(pkl_path, msk_path):
    pkl_data = load_pkl(pkl_path)   # 获取整个pkl而不是其中某一个数组
    drone_disp = read_msk_pkl(msk_path,'disp')[::2]   #无人机的视差
    
    if pkl_data is None or drone_disp is None:
        return None
    
    msk = read_msk_pkl(msk_path,'mask')[::2]    # 获取mask
    # msk小一点，因为rgb图有抗锯齿，和mask不匹配
    msk = dilation_msk(msk)
    
    # 修改pkl中的视差图
    ori_disp = decompress_pkl(pkl_data['left_disparity'])
    comb_data = ori_disp.copy()
    comb_data[msk] = drone_disp[msk]
    pkl_data['left_disparity']['data'] = lz.compress(comb_data, 3) 
    pkl_data['has_drone_msk'] = True    # 标记    
    save_pkl(pkl_data,pkl_path)

    # plt.figure()
    # plt.imshow(comb_data, cmap='jet')
    #return pkl_data

def combine_seg_pkl(pkl_path, msk_path):
    pkl_data = load_pkl(pkl_path)   # 获取整个pkl而不是其中某一个数组
    drone_seg = read_msk_pkl(msk_path,'seg')[::2]   #无人机的视差
    
    if pkl_data is None or drone_seg is None:
        return None
    
    msk = read_msk_pkl(msk_path,'mask')[::2]    # 获取mask
    msk = dilation_msk(msk)
    # 修改pkl中的视差图
    ori_seg = decompress_pkl(pkl_data['segmentation'])
    comb_data = ori_seg.copy()
    comb_data[msk] = drone_seg[msk]
    pkl_data['segmentation']['data'] = lz.compress(comb_data, 3)
    pkl_data['has_drone_msk'] = True    # 标记
    save_pkl(pkl_data, pkl_path)

def combine_img_pkl(pkl_path, msk_path):
    '''左图右图都要加上msk'''
    pkl_data = load_pkl(pkl_path)   # 获取整个pkl而不是其中某一个数组
    if pkl_data is None:
        return None
    if 'right' in msk_path:
        msk_path = msk_path.replace('right', 'left')
    # 合并左图
    limg = decompress_pkl(pkl_data['left_image'])
    lmsk = read_msk_pkl(msk_path,'mask')[::2]
    lmsk = dilation_msk(lmsk)
    drone_limg = read_msk_pkl(msk_path, 'rgb')[::2]
    comb_limg = limg.copy()
    comb_limg[lmsk] = drone_limg[lmsk]
    
    # 合并右图
    rimg = decompress_pkl(pkl_data['right_image'])
    rmsk = read_msk_pkl(msk_path.replace('left','right'),'mask')[::2]
    rmsk = dilation_msk(rmsk)
    drone_rimg = read_msk_pkl(msk_path.replace('left','right'),'rgb')[::2]
    comb_rimg = rimg.copy()
    comb_rimg[rmsk] = drone_rimg[rmsk]
    
    # 重写pkl
    pkl_data['left_image']['data'] = lz.compress(comb_limg, 3)
    pkl_data['right_image']['data'] = lz.compress(comb_rimg, 3)
    pkl_data['has_drone_msk'] = True
    save_pkl(pkl_data, pkl_path)
    # plt.figure()
    # plt.imshow(comb_limg, cmap='jet')
    # plt.figure()
    # plt.imshow(comb_rimg, cmap='jet')
    # plt.show()    
    # return pkl_data

def process_pkl_folder(dispfolder):
    '''dispfolder:以视差路径为主路径'''
    # 以视差路径为主路径，只要下视
    down_path = [str(p) for p in Path(dispfolder).rglob('cam0_0') if p.is_dir()][0]
    disp_files = [str(p) for p in Path(down_path).rglob('*.pkl') if p.is_file()]
        
    with Pool(cpu_count()-10) as p:
        p.map(partial(process_one), disp_files)    

def process_one(disp_path):
    seg_path = disp_path.replace('Disparity', 'Segment')
    img_path = disp_path.replace('Disparity', 'Images')
    msk_file = random.choice(msk_files)
    if 'right' in msk_file:
        msk_file =  msk_file.replace('right', 'left')    
        
    combine_disp_pkl(disp_path, msk_file)
    combine_seg_pkl(seg_path, msk_file)
    combine_img_pkl(img_path, msk_file)
    
    _logger.info(f'processed {disp_path}')
            
if __name__ == "__main__":
    folder_path = '/mnt/119-data/R22612/Data/ERP/train/synthesis'
    # folders = glob.glob(folder_path+'/*')
    # tiny_folders = [str(p) for p in folders if 'hypertiny' in p]
    # normal_folders = [str(p) for p in folders if 'tiny' not in p]
    # #random.shuffle(tiny_folders)
    # #print(tiny_folders)
    # for tf in tiny_folders[:20]:
    #     print(tf)
    #     process_pkl_folder(tf+'/Disparity')
    # for nf in normal_folders[:20]:
    #     print(nf)
    #     process_pkl_folder(nf+'/Disparity')
    #process_pkl_folder(folder_path)
    pkl_path = "\\10.250.6.119\mnt-data\R22612\Data\ERP\train\synthesis\add_2x_substation_2023-11-08-12-25-01_synthesis\Images\2x_substation_2023-11-08-12-25-01_out_renamed\group0\cam0_0\Image_erp\1699406779391984128.pkl"
    show_pkl(pkl_path,'left_image')
    #combine_img_pkl(pkl_path, random.choice(msk_files))