import cv2
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import lz4.frame as lz
import pickle as plk
import os

# 根据深度提取细小物体
def get_thinmask_by_depth(depth, kernel, far_mask, threshold):
    depth = depth * 1   # 传入的depth是只读的，这一步使depth可写
    closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    diff = np.abs(depth - closing)
    thin_mask = diff > threshold
    thin_mask[far_mask] = False
    # thin_mask_clean = remove_large_and_small_connected_components(thin_mask, 15, 1000)
    
    # plt.figure()
    # plt.imshow(depth[1280:,:], cmap='viridis_r', vmax = 300, interpolation='nearest')
    # plt.figure()
    # plt.imshow(diff[1280:,:], cmap='viridis', interpolation='nearest')
    # #plt.colorbar()  # Add a colorbar for reference
    # plt.figure()
    # plt.imshow(thin_mask[1280:,:])

    # use thin_mask on depth
    depth[thin_mask] = 0
    plt.figure()
    plt.imshow(depth[1280:,:], cmap='viridis_r', vmax = 40, interpolation='nearest')    
    
    plt.show()
    
    return thin_mask    

# 根据视差提取细小物体
def get_thinmask_by_disp(disp, kernel, far_mask, threshold):
    opening = cv2.morphologyEx(disp, cv2.MORPH_OPEN, kernel)
    diff = np.abs(disp - opening)
    thin_mask = diff > threshold
    thin_mask[far_mask] = False 
    
    plt.figure()
    plt.imshow(disp[1280:,:])
    plt.figure()
    plt.imshow(thin_mask[1280:,:])
    plt.show()

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

def remove_large_and_small_connected_components(image, min_size, max_size):
    # # 将图像转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化图像
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    binary = image.astype(np.uint8)
    
    # 执行连通组件标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 创建一个空白的图像，用于存储结果
    result = np.zeros_like(image)

    # 循环处理每个连通域
    for label in range(1, num_labels):
        # 获取当前连通域的面积
        area = stats[label, cv2.CC_STAT_AREA]
        
        # 检查面积是否在指定范围内
        if min_size <= area:# <= max_size:
            # 如果面积在范围内，则将连通域复制到结果图像中
            result[labels == label] = image[labels == label]

    return result

#disp = np.load(r"D:\Codes\ERPvs123\eval_data\Autel_Evening_Highway_2023-08-08-17-49-16_out\group1\cam1_0\Disparity\1691488148759.npz")['data'].astype(np.float32)
#path = r"Y:\simulation\train\2x\2x_Autel_Forest\2x_Autel_Forest_2023-08-28-10-57-25\cube_front\CubeDepth\1693191405399024640.lz4"

folder_path = r"Y:\simulation\train\tiny_test_data"
far_value = 30  # 只关心30m以内的细小物体
threshold = 0.7 # 分辨细小物体边界的阈值，反复测试确定
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))    # 用于膨胀腐蚀的核，细小物体的定义为8个像素，故核的大小为9


lz4_files = [str(p) for p in Path(folder_path).rglob("*.lz4") if p.is_file()]
for f in lz4_files:
    depth = read_lz4(f)
    far_mask = depth > far_value
    get_thinmask_by_depth(depth, kernel, far_mask, threshold)
    