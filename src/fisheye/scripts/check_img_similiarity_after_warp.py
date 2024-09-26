# This file is written by Xuanquan.
from skimage import io, feature
#from scipy import ndimage
import numpy as np
import time
import cv2
import os
import lz4.frame as lz
import pickle as plk
from loguru import logger
from pathlib import Path
import sys
from numba import njit
import torch
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial

dir2 = r"D:\simudata\DataBackup\Autel_Japanese_Street_2023-08-09-19-08-11\cube_rear\CubeScene\1691578471759.png"
dir3 = r"D:\simudata\DataBackup\Autel_abandonedfactory_2023-08-08-15-11-51\cube_front\CubeScene\1691478710805"
#定义阈值
t_s = 0.8   #判定为像素相似的阈值
t_rate = 0.7    #判定相似度百分比的阈值
t_mean = 0.7    #均值法求相似的阈值

#定义一个进度条
def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()

#读取lz4，目的是读取视差图
def read_lz4(lz4_file):
    if os.path.exists(lz4_file):
        with open(lz4_file, 'rb') as f:
            cdata = plk.load(f)
        arr = lz.decompress(cdata['arr'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {lz4_file}')
        return arr  

# 基于视差和左右视重建
def rebuild_left(image_R, disp_L):   
    tar_img = torch.tensor(image_R).permute(2,0,1).unsqueeze(0).float()
    ref_recon, mask_range = warp(tar_img, -disp_L)
    
    ref_recon = ref_recon.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    
    return ref_recon

def occ_mask(disp_ref):
    B, C, H, W = disp_ref.size()
    assert B==1 and C == 1
    # mesh grid
    xx = torch.arange(0, W, device=disp_ref.device).view(1, -1).repeat(H, 1)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    xx = torch.round(xx + disp_ref)
    disp_tar = torch.full((B, C, H, W), float('nan'))
    disp_project(disp_ref, H, W, xx, disp_tar)
    disp_recon, mask = warp(disp_tar, disp_ref)
    mask_occ = (disp_recon-disp_ref).abs()<1.0
    mask_occ = torch.logical_or(mask_occ, torch.isnan(disp_tar))
    return mask_occ

# @njit
def disp_project(disp_ref, H, W, xx, disp_tar):
    for h in range(H):
        for w in range(W):
            w_tar = int(xx[0,0,h,w])
            if 0 <= w_tar <= W-1: 
                if torch.isnan(disp_tar[0,0,h,w_tar]):
                    disp_tar[0,0,h,w_tar] = disp_ref[0,0,h,w]
                elif disp_ref[0,0,h,w].abs() > disp_tar[0,0,h,w_tar].abs():
                    disp_tar[0,0,h,w_tar] = disp_ref[0,0,h,w]

def verify_disparity(image_L, image_R, disp_L,img_name,recon_path):
    ref_img = torch.tensor(image_L).permute(2,0,1).unsqueeze(0).float()
    tar_img = torch.tensor(image_R).permute(2,0,1).unsqueeze(0).float()
    ref_recon, mask_range = warp(tar_img, -disp_L)
    img_path = os.path.join(recon_path, img_name)
    ref_recon = ref_recon.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    cv2.imwrite(img_path,ref_recon)
    return ref_recon

def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the disparity
    x: [B, C, H, W] (im2)
    disp: [B, 1, H, W] disparity in the view of im1, negative
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1) + 0.5
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1) + 0.5
    vgrid = torch.cat((xx, yy), 1).float()

    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] + disp
    mask = torch.logical_and(vgrid[:,:1,:,:] >= 0, vgrid[:,:1,:,:] <= W - 1)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x.float(), vgrid)
    return output, mask

@njit    # 计算相关性相似度，@njit 可以加速数学计算，原来6分钟的计算过程现在只要十几秒
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


'''
im1,im2:要对比的图片
d:pixel patch的extend，图片分辨率较大时，d应该越大，比如三倍分辨率下d推荐为3，最小为1
sample:为了加快计算可以对图片抽样计算，一般几倍分辨率就抽样几倍，不抽样则为1
'''
def compare(im1, im2, d=1, sample=1, show=False, timebar=True):
    #im1 = io.imread(name + '.png', as_gray=True)
    #im2 = io.imread(name + '.webp', as_gray=True)
    similar, total = 0,0
    
    sh_row, sh_col= im1.shape[0], im1.shape[1]

    correlation = np.zeros((int(sh_row/sample), int(sh_col/sample)))
    total = correlation.shape[0] * correlation.shape[1]

    t = time.perf_counter()
    #print(t)
    x,y=0,0     #correlation自己的当前像素坐标
    for i in range(d, sh_row - (d + 1), sample):
        if timebar:
            process_bar(x+1, correlation.shape[0])
        y = 0
        for j in range(d, sh_col - (d + 1), sample):
            
            correlation[x, y] = correlation_coefficient(im1[i - d: i + d + 1,
                                                            j - d: j + d + 1],
                                                        im2[i - d: i + d + 1,
                                                            j - d: j + d + 1])
            # 统计相似度
            if correlation[x, y] > t_s:
                similar += 1
            
            y += 1
        x += 1
            #print(f'{i},{j} cost:{time.perf_counter() - t}')

    print(f' 耗时{time.perf_counter() - t}s')
    if show:
        io.imshow(correlation, cmap='gray')
        io.show()
    
    return float(similar/total)

'''
folder_path：文件夹路径
sample:文件采样比例，0-1之间，推荐0.1
'''
def process_dataset(folder_path, sample=0.1):
    # 根据文件夹路径提取左右和视差图
    print('检查针孔图') 
    imgL, dispL, imgR = get_file_list(folder_path,'group0')
    subprocess_dataset(imgL, dispL, imgR, sample)
    print('检查全景图')
    imgL, dispL, imgR = get_file_list(folder_path,'group1')
    subprocess_dataset(imgL, dispL, imgR, sample)
    print('检查1拆3')
    imgL, dispL, imgR = get_file_list(folder_path,'group1',True)
    subprocess_dataset(imgL, dispL, imgR, sample)


def subprocess_dataset(imgL, dispL, imgR, sample):
    count = int(len(imgL) * sample)       # 采样范围
    imgL = imgL[:count]
    dispL = dispL[:count]
    imgR = imgR[:count]
    
    pairs = (imgL, imgR, dispL)
    pairs = list(zip(*pairs))
    # for i in range(count):
    #     check_rebuilt(imgL[i], imgR[i], dispL[i])
    # 多线程批处理
    with Pool(multiprocessing.cpu_count()-1) as p:
        p.map(check_rebuilt_thread, pairs)

# pair是左视图-右视图-左视差的集合        
def check_rebuilt_thread(pair):
    # 读取图片
    img_name= os.path.basename(pair[0])
    imgL = cv2.imread(pair[0])
    imgR = cv2.imread(pair[1])
    dispL = read_lz4(pair[2])
    # 重建左视图
    ref = rebuild_left(imgR, dispL)
    # 比较重建前后
    res = compare(imgL, ref, timebar=True)
    print(f'similarity of {img_name} is {res}')   

def check_rebuilt(pathL,pathR,pathD):
    # 读取图片
    img_name= os.path.basename(pathL)
    imgL = cv2.imread(pathL)
    imgR = cv2.imread(pathR)
    dispL = read_lz4(pathD)
    # 重建左视图
    ref = rebuild_left(imgR, dispL)
    # 比较重建前后
    res = compare(imgL, ref)
    print(f'similarity of {img_name} is {res}')     

def check_rebuilt_by_mean(pathL,pathR,pathD):
    # 读取图片
    img_name= os.path.basename(pathL)
    imgL = cv2.imread(pathL)
    imgL = np.array(imgL, dtype= np.int16)
    imgR = cv2.imread(pathR)
    dispL = read_lz4(pathD)
    # 重建左视图
    ref = rebuild_left(imgR, dispL)
    # occ
    mask = occ_mask(torch.tensor(dispL[None,None,...]).float()).squeeze().unsqueeze(-1).repeat(1,1,3).numpy()
    
    ref = np.array(ref, dtype= np.float32)
    ref[mask] = np.nan
    # 比较重建前后
    res = np.nanmean(np.abs(ref - imgL))
    #cv2.imshow(np.abs(ref - imgL).astype(np.uint8))
    print(f'similarity of {img_name} is {res}')  

def get_file_list(filepath, folder, _3to1 = False):
    img_folder = 'Image'
    disp_folder = 'Disparity'
    if _3to1:
        img_folder = 'Image_3to1'
        disp_folder = 'Disp_3to1'
    left_fold = os.path.join(filepath, folder,'cam'+folder[5:]+'_0',img_folder)
    disp_L_fold = os.path.join(filepath, folder,'cam'+folder[5:]+'_0',disp_folder)
    right_fold = os.path.join(filepath, folder,'cam'+folder[5:]+'_1',img_folder)

    # 如果不存在视差目录，则返回None
    if not os.path.exists(disp_L_fold):
        logger.error('不存在视差目录')
        return None, None, None, None

    image_L = [img for img in os.listdir(left_fold) if img.endswith('.webp')]
    disp_L = [img for img in os.listdir(disp_L_fold) if img.endswith('.lz4')]
    image_R = [img for img in os.listdir(right_fold) if img.endswith('.webp')]
    
    if len(disp_L)==0:
        logger.error('没有识别到视差图，检查视差图是否是lz4格式')
        return None, None, None, None

    image_L.sort()
    disp_L.sort()
    image_R.sort()

    image_L = [os.path.join(left_fold, img) for img in image_L]
    disp_L = [os.path.join(disp_L_fold, img) for img in disp_L]
    image_R = [os.path.join(right_fold, img) for img in image_R]

    return image_L, disp_L, image_R    

if __name__ == '__main__':
    root_path = r"D:\simudata\DataBackup\Autel_Demo_Map_2023-08-08-15-37-32_out"   
    process_dataset(root_path, 0.1)

# imgL = cv2.imread(root_path + r"\group0\cam0_0\Image\1691480249638.webp")
# imgR = cv2.imread(root_path + r"\group0\cam0_1\Image\1691480249638.webp")
# disL = read_lz4(root_path + r"\group0\cam0_0\Disparity\1691480249638.lz4")
    
# ref = verify_disparity(imgL, imgR, disL, 'rebuild.png', root_path)

# cv2.imshow("ref",ref)
# cv2.imshow("ref", np.abs(ref.astype(np.float32)-imgL.astype(np.float32)).astype(np.uint8))
# #cv2.imshow('l', imgL)
# cv2.waitKey(0)
# compare(imgL, ref, show=True)