import cv2
import os
#from matplotlib import pyplot as plt
import numpy as np

#import torch
#import imageio

import cv2
import numpy as np
from numba import njit 
import pickle as pkl
import lz4.frame as lz
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

import math
import run_panorama_disparity_from_depth
import run_pinholes_disparity_from_depth
import panorama_to_erp
import simu_params
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 一些全局变量默认值
g_cubedisp_name = 'CubeDisparity_0.105'
g_down_baseline = 0.105
g_pinhole_baseline = 0.06
g_focallength = 1446.238224784178

def get_occlusion_mask(shifted):
    """
    shifted: [H, W] sample_grid in the view of im1
    """
    h, w = shifted.shape[:2]

    mask_up = shifted > 0
    mask_down = shifted > 0

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)

    for col in range(w - 2):
        loc = shifted[:, col:col + 1]  # keepdims
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
        (shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
        (shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    # mask_less_occ = mask_up + mask_down
    mask_more_occ = mask_up * mask_down

    return mask_more_occ


def warp_cv(x, disp):
    """
    warp an image (im2) back to im1, according to the disparity
    x: [H, W, C] (im2)
    disp: [H, W] disparity in the view of im1, negative
    """
    #logger.info('start warp_cv')
    
    H, W, C = x.shape

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # Add disparity to the x coordinates
    xx_disp = xx + disp

    # Get occlusion mask
    #logger.info('get occlusion mask')
    mask_more_occ = get_occlusion_mask(xx_disp)


    # Generate the valid mask
    mask = (xx_disp >= 0) & (xx_disp <= W - 1)

    # Clip the coordinates for cv2.remap
    #logger.info('clip the coordinates')
    xx_disp = np.clip(xx_disp, 0, W-1)

    # Initialize the output image
    #logger.info('remap the output image')
    warped_image = cv2.remap(x, xx_disp, yy, interpolation=cv2.INTER_LINEAR)

    #logger.info('end warp_cv')
    return warped_image, mask * mask_more_occ

@njit    # Thanks to Xuanquan, njit 可以加速数学计算，原来6分钟的计算过程现在只要十几秒
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    product /= stds
    return product

def compare(im1, im2, d=9, sample=9):
    #logger.info('start compare')
    similar, total = 0, 0
    sh_row, sh_col = im1.shape[0], im1.shape[1]
    correlation = np.zeros((int(sh_row/sample), int(sh_col/sample)))
    total = correlation.shape[0] * correlation.shape[1]
    for x, i in enumerate(range(d, sh_row - (d + 1), sample)):
        for y, j in enumerate(range(d, sh_col - (d + 1), sample)):
            correlation[x, y] = correlation_coefficient(im1[i - d: i + d + 1,
                                                            j - d: j + d + 1],
                                                        im2[i - d: i + d + 1,
                                                            j - d: j + d + 1])
            if correlation[x, y] > 0.8:   # Threshold to determine a pixel is similar or not
                similar += 1
    #logger.info('end compare')
    return float(similar/total)

def dilation(dpdata, d = 1):
    
    #expand = np.zeros_like(dpdata, dtype = np.float32)
    expand = dpdata.copy()
    
    # 膨胀特征
    d=1 + 2*d
    kernel = np.ones((d,d), np.uint8)
    # 实际操作是反向腐蚀，因为要让近处的深度覆盖远处的，即值小的优先
    #expand = cv2.dilate(expand, kernel, 1)
    expand = cv2.erode(expand, kernel, 1)   
        
    return expand

def save_gif(left, right, disp_gt, out_path, uncert_invalid=None, uncert_grad_invalid=None, border=0):
    if uncert_invalid is None:
        uncert_invalid = np.zeros_like(disp_gt)>0
    if uncert_grad_invalid is None:
        uncert_grad_invalid = np.zeros_like(disp_gt)>0
    # image text configure
    font                   = cv2.FONT_HERSHEY_DUPLEX
    LeftTopCornerOfText    = (10, 30)
    LeftBottomCornerOfText = (10, left.shape[-2]-10)
    fontScale              = 1
    fontColor              = (0, 255, 0)
    thickness              = 1
    lineType               = cv2.LINE_AA
    # left, right: [3, H, W]; disp, disp_gt: [H,W].
    assert left.shape == right.shape, "The input images have inconsistent shapes."

    in_h, in_w = left.shape[-2:]
    left_np = left#.permute(1,2,0).cpu().numpy().astype(dtype='uint8').copy()
    right_np = right#.permute(1,2,0).cpu().numpy().astype(dtype='uint8').copy()
    left_np = cv2.cvtColor(left_np, cv2.COLOR_BGR2RGB)
    right_np = cv2.cvtColor(right_np, cv2.COLOR_BGR2RGB)

    disp_gt_max = np.nanmax(disp_gt) if border<=0 else disp_gt[border:-border, :].max()
    disp_max = disp_gt_max
    disp_gt_min = np.nanmin(disp_gt)
    disp_min = disp_gt_min

    # disparity groundtruth color map
    # disp_max=1.25
    disp_gt_vis = (disp_gt - disp_min) / (disp_max - disp_min) * 255.0
    #disp_gt_vis = disp_gt_vis.clamp(min=0, max=255.0).cpu().numpy().astype("uint8")
    disp_gt_vis = np.clip(disp_gt_vis, 0, 255).astype(np.uint8)
    
    disp_gt_vis = cv2.applyColorMap(disp_gt_vis, 12)#cv2.COLORMAP_INFERNO)
    disp_gt_vis_ori = disp_gt_vis.copy()
    disp_gt_vis[uncert_invalid] = (disp_gt_vis[uncert_invalid]*0.4 + 0*0.6).astype(np.uint8) # color map 变暗
    disp_gt_vis[uncert_grad_invalid] = (disp_gt_vis[uncert_grad_invalid]*0.4 + 255*0.6).astype(np.uint8) # color map 变亮
    overlap_mask = uncert_invalid & uncert_grad_invalid
    disp_gt_vis[overlap_mask] = 0 # 黑色
    # disp_gt_vis[disp_gt<=0] = 0
    disp_gt_vis = cv2.putText(disp_gt_vis, f'disp_gt', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    disp_gt_vis = cv2.putText(disp_gt_vis, f'max={disp_gt_max:.2f}', LeftBottomCornerOfText, font, fontScale, fontColor, thickness, lineType)
    disp_gt_vis_ori = cv2.putText(disp_gt_vis_ori, f'disp_gt', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    disp_gt_vis_ori = cv2.putText(disp_gt_vis_ori, f'max={disp_gt_max:.2f}', LeftBottomCornerOfText, font, fontScale, fontColor, thickness, lineType)

    # reconstructed ground truth left image
    # left_recon_gt, mask_range = warp(right.reshape(1,-1,in_h,in_w), -disp_gt.reshape(1,1,in_h,in_w))
    #left_recon_gt_cv, mask_range_cv = warp_cv(right.permute(1,2,0).numpy(), -disp_gt.numpy())
    left_recon_gt_cv, mask_range_cv = warp_cv(right, -disp_gt)
    # mask_occ = occ_mask(-disp.reshape(1,1,in_h,in_w).cpu())

    # left_recon_gt_mask = left_recon_gt * mask_range #* mask_occ
    # left_recon_gt_mask = left_recon_gt_mask.reshape(-1,in_h,in_w).permute(1,2,0).cpu().numpy().astype(dtype='uint8').copy()
    # left_recon_gt_mask = cv2.cvtColor(left_recon_gt_mask, cv2.COLOR_BGR2RGB)
    # left_recon_gt_mask[disp_gt<0] = 0
    # left_recon_gt_mask = cv2.putText(left_recon_gt_mask, 'left_recon_gt', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    
    left_recon_gt_mask_cv = left_recon_gt_cv * np.expand_dims(mask_range_cv, axis=-1) #* mask_occ
    # left_recon_gt_mask_cv = left_recon_gt_mask_cv.reshape(-1,in_h,in_w).permute(1,2,0).cpu().numpy().astype(dtype='uint8').copy()
    left_recon_gt_mask_cv = cv2.cvtColor(left_recon_gt_mask_cv, cv2.COLOR_BGR2RGB)
    left_recon_gt_mask_cv[disp_gt<0] = 0
    left_recon_gt_mask_cv = cv2.putText(left_recon_gt_mask_cv, 'left_recon_gt', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)

    # put text
    left_np = cv2.putText(left_np, 'left', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    right_np = cv2.putText(right_np, 'right', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)

    # Concat
    all_vis = np.concatenate((right_np,left_recon_gt_mask_cv,disp_gt_vis_ori,left_recon_gt_mask_cv), axis=1)
    ref_vis = np.concatenate((left_np,left_np,left_np,left_recon_gt_mask_cv), axis=1)
    gif_list = [all_vis, ref_vis]

    # save gif
    parent_path = os.path.abspath(os.path.join(out_path, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    imageio.mimsave(out_path, gif_list, duration=[500,500], loop=0)

    cv2.imwrite(os.path.join(parent_path,'tmp1.png'), all_vis)
    cv2.imwrite(os.path.join(parent_path,'tmp2.png'), ref_vis)       

def read_lz4(lz4_file):
    if os.path.exists(lz4_file):
        with open(lz4_file, 'rb') as f:
            cdata = pkl.load(f)
        arr = lz.decompress(cdata['arr'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {lz4_file}')
        return arr     

def remove_curframe(file_dir):
    '''
    移除当前帧，即该帧持有的rgb,分割，视差图
    '''
    if 'cube_front' in file_dir:
        split_name='cube_front'
    elif 'left' in file_dir:
        split_name='left'
    else:
        print(f'Error: {file_dir} not in cube_front or left')
        return
    
    root_dir = str(Path(file_dir).parent).split(split_name)[0]
    name = Path(file_dir).name.split('.')[0]
    rm_files = [str(p) for p in Path(root_dir).rglob(f'{name}*') if p.is_file()]
    for rm_file in rm_files:
        os.remove(rm_file)
    print(f'Remove {name} done')

def _pinhole_depth2disp(depth_file,baseline = 0.06,focal_length = 1446.238224784178):

    depth_data = read_lz4(depth_file)
    #print(f'baseline={baseline}, focal_length={focal_length}')
    
    disparity = baseline * focal_length / depth_data
    disparity = disparity.astype(np.float16)

    return disparity

def process_compare(img_L, img_R, disp_L):
    '''
    img_L, img_R, disp_L均为np数组，并且尺寸相同
    '''
    if (img_L.shape[:2] != disp_L.shape[:2] or
        img_R.shape[:2] != disp_L.shape[:2] or
        img_L.shape != img_R.shape):
        return 0
    
    disp_L = disp_L.astype(np.float32)
    disp_L[~((disp_L>=0) & (disp_L<256))] = np.nan  # 过近的视差对匹配无帮助，剔除

    re_limg, mask = warp_cv(img_R, -disp_L) # 重建左图
    # 膨胀mask,有助于消除细小物体视差膨胀造成的误差
    mask=mask.astype(np.uint8)
    mask = dilation(mask, 1)
    mask = mask>0    

    re_limg[~mask] = 0
    img_L[~mask] = 0
    # 比较差异
    return compare(re_limg,img_L)

def process_compare_by_cubedisp(cubedisp_dir):
    # 根据鱼眼视差获取对应左右RGB图
    disp_L = read_lz4(cubedisp_dir)
    img_L_path = cubedisp_dir.replace(g_cubedisp_name, 'CubeScene').replace('lz4','webp')
    img_L = cv2.imread(img_L_path)
    img_R_path = img_L_path.replace('cube_front','cube_below')
    img_R = cv2.imread(img_R_path)
    
    if img_L is None or img_R is None or disp_L is None:
        remove_curframe(cubedisp_dir)
        return

    similarity = process_compare(img_L[::2,::2], img_R[::2, ::2], disp_L[::2, ::2]*0.5) # 降采样加快速度，注意视差降采样后要乘以相应倍数
    if similarity < 0.6:
        # 使用深度图进行二次检测
        cube_depth_path = cubedisp_dir.replace(g_cubedisp_name, 'CubeDepth').replace('train_out','train')
        new_disp_L = run_panorama_disparity_from_depth._depth2disp(cube_depth_path, '', g_down_baseline, dilation_tiny=False, save_disp=False)
        new_disp_resize = panorama_to_erp.crop_erp(new_disp_L, is_disp=True, is_seg_gray=False, data_dir=cube_depth_path)    # resize精确视差图
        new_similarity = process_compare(img_L[::2,::2], img_R[::2, ::2], new_disp_resize[::2, ::2]*0.5)
        logger.info(f'{cubedisp_dir} need second check: simlarity={similarity:.2f}, second similarity={new_similarity:.2f}')
        if new_similarity < 0.6:            
            # 移除这一帧
            remove_curframe(cubedisp_dir)
    logger.info(f'{cubedisp_dir} compare done')
    
def process_compare_by_pinhole_disp(pinhole_disp_dir):
    # 根据针孔视差获取对应左右RGB图
    disp_L = read_lz4(pinhole_disp_dir)
    img_L_path = pinhole_disp_dir.replace('Disparity','Scene').replace('lz4','webp')
    img_L = cv2.imread(img_L_path)
    img_R_path = img_L_path.replace('left','right')
    img_R = cv2.imread(img_R_path)
    
    if img_L is None or img_R is None or disp_L is None:
        remove_curframe(pinhole_disp_dir)
        return
    
    similarity = process_compare(img_L[::2,::2], img_R[::2, ::2], disp_L[::2, ::2]*0.5) # 降采样加快速度，注意视差降采样后要乘以相应倍数
    if similarity < 0.6:
        # 使用深度图进行二次检测
        #print(f'similarity={similarity:.2f}, second check:{pinhole_disp_dir}')
        pinhole_depth_path = pinhole_disp_dir.replace('Disparity', 'DepthPlanar').replace('train_out','train')
        args = simu_params.ParaPinhole(g_focallength, g_pinhole_baseline)
        new_disp_L = run_pinholes_disparity_from_depth._depth2disp(pinhole_depth_path, '', args, dilation_tiny=False, save_disp=False)
        new_similarity = process_compare(img_L[::2,::2], img_R[::2, ::2], new_disp_L[::2, ::2]*0.5)
        logger.info(f'{pinhole_disp_dir} need second check: simlarity={similarity:.2f}, second similarity={new_similarity:.2f}')
                   
        if new_similarity < 0.6:            
            remove_curframe(pinhole_disp_dir)
    
    logger.info(f'{pinhole_disp_dir} compare done')
            
def multiprocess_compare_after_resize(folder_dir, down_baseline=0.105, pinhole_baseline=0.06, focallength=1446.238224784178):
    '''
    在resize之后根据右图和视差重建左图，比较左图和原左图的相似度，以确保视差正确有效
    由于视差经过了膨胀可能不够准确，所以使用2次自检，首次直接用视差自检初筛，未通过的再用深度进行二次自检
    2次自检的意义在于，只有有问题的数据才使用深度图，减少计算量
    
    folder_dir：基本数据集目录，通常格式为2x_机型_场景_时间
    down_baseline:鱼眼下视baseline, 自检用下视的baseline,因为下视baseline比较固定，不经常改
    '''
    # 检查是否已经自检过
    if os.path.exists(Path(folder_dir) / 'check.log'):
        logger.info(f'{folder_dir} already checked')
        return
    logger.info(f'start check {folder_dir}')
    g_down_baseline = down_baseline
    g_pinhole_baseline = pinhole_baseline
    g_focallength = focallength
    
    # 自检全景图
    g_cubedisp_name = f'CubeDisparity_{str(down_baseline)}'    
    cubedisp_root = Path(folder_dir) / 'cube_front' / g_cubedisp_name
    cube_disp = [str(p) for p in Path(cubedisp_root).rglob('*.lz4') if p.is_file()]
    # for disp in cube_disp:
    #     process_compare_by_cubedisp(disp)
    with Pool(cpu_count()-1) as p:
        p.map(process_compare_by_cubedisp, cube_disp)
    logger.info('check cube_front done')
    # 自检针孔图
    pinhole_root = Path(folder_dir) / 'left' / 'Disparity'
    pinhole_disp = [str(p) for p in Path(pinhole_root).rglob('*.lz4') if p.is_file()]
    with Pool(cpu_count()-1) as p:
        p.map(process_compare_by_pinhole_disp, pinhole_disp)
    logger.info('check pinhole done')      
    
    with open(Path(folder_dir) / 'check.log', 'a') as f:
        f.write(f'{folder_dir} done\n')      
                
if __name__ == '__main__':
    import pickle
    import numpy as np
    # from lz4.frame import compress as lzcompress
    # from lz4.frame import decompress as lzdecompress
    import csv
        
    def image_to_tensor(image_path):
        # if '/mnt/' in image_path:
        #     image_path = image_path.replace('/mnt/119-data',r'\\10.250.6.119')
        #     image_path = image_path.replace('/','\\')    # 读取图片
        image = cv2.imread(image_path)
        # 将图片转换为RGB格式
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图片转换为np
        tensor = np.array(image)
        return tensor
    
    d = '/mnt/119-data/samba-share/simulation/evalue'
    folders = [str(p) for p in Path(d).rglob('2x*') if p.is_dir()]
    for f in folders:
        multiprocess_compare_after_resize(f, 0.105, 0.06)
        
    # (r"D:\train_out\2x_MX128_bigCity_2024-05-15-17-23-16\cube_front\CubeDisparity_0.128\1715764973030.lz4")