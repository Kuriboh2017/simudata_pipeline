import numpy as np
import numba
import logging

# 将numba的日志等级提高到warning，从而屏蔽njit的超长log
logging.getLogger('numba').setLevel(logging.WARNING)

class Resize:
    def __init__(self, in_height=None, in_width=None, out_height=None, out_width=None, interpolation=None, ipsize=None):
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = out_height
        self.out_width = out_width
        self.interpolation = interpolation
        
        if None not in [in_height, in_width, out_height, out_width, interpolation, ipsize]:
            # 计算垂直和水平方向的缩放因子
            scale_factor_y = out_height / in_height
            scale_factor_x = out_width / in_width

            # 生成新图像的网格坐标
            y = np.arange(0, out_height).reshape(-1, 1).repeat(out_width, axis=1)
            x = np.arange(0, out_width).reshape(1, -1).repeat(out_height, axis=0)

            # 将网格坐标映射回原始图像坐标
            y = (y + 0.5) / scale_factor_y - 0.5
            x = (x + 0.5) / scale_factor_x - 0.5

            # 提取坐标整数部分
            y_int = np.floor(y).astype(int)
            x_int = np.floor(x).astype(int)

            # 对四个最近邻像素坐标做越界处理，确保不超出图像范围
            # self.y_int_clip = np.clip(y_int, 0, in_height - 1)
            # self.x_int_clip = np.clip(x_int, 0, in_width - 1)
            # self.y_int_1_clip = np.clip(y_int + 1, 0, in_height - 1)
            # self.x_int_1_clip = np.clip(x_int + 1, 0, in_width - 1)
            
            lens = ipsize*2
            # 对插值范围做越界处理
            self.y_range = np.empty(lens, dtype=object)
            self.x_range = np.empty(lens, dtype=object)
            for i in range(lens):
                self.y_range[i] = np.clip(y_int - ipsize + 1 + i, 0, in_height - 1)
                self.x_range[i] = np.clip(x_int - ipsize + 1 + i, 0, in_width - 1)
                
    def resize(self, image, out_height=None, out_width=None, interpolation=None, ipsize=1):
        # 输入：image - 输入图像数组，out_height和out_width - 目标图像的高度和宽度
        # interpolation: 'min' or 'max' or 'bilinear'

        # 获取原始图像的高度和宽度
        in_height, in_width = image.shape[:2]
        if interpolation is None: interpolation = self.interpolation
        if out_height is None: out_height = self.out_height
        if out_width is None: out_width = self.out_width

        if self.in_height != in_height or self.in_width != in_width \
            or self.out_height != out_height or self.out_width != out_width \
            or self.interpolation != interpolation:
            #print('init resize')
            self.__init__(in_height, in_width, out_height, out_width, interpolation, ipsize)

        # 计算四个最近邻像素值
        # Q11 = image[self.y_int_clip, self.x_int_clip]
        # Q12 = image[self.y_int_clip, self.x_int_1_clip]
        # Q21 = image[self.y_int_1_clip, self.x_int_clip]
        # Q22 = image[self.y_int_1_clip, self.x_int_1_clip]
        
        # 计算插值范围像素值
        new_image = self.cauculate_img(self.y_range,self.x_range,image, ipsize,interpolation)
        # elif interpolation == 'bilinear':
        #     new_image = self.bilinear(Q11, Q21, Q12, Q22, self.y_frac, self.x_frac)

        return new_image

    def cauculate_img(self, y_range,x_range,image, ipsize,interpolation):
        new_image = image[y_range[0], x_range[0]]
        for i in range(ipsize*2):
            for j in range(ipsize*2):
                if interpolation == 'min':
                    new_image = np.minimum(new_image, image[y_range[i], x_range[j]])
                elif interpolation == 'max':
                    new_image = np.maximum(new_image, image[y_range[i], x_range[j]]) 
            
        return new_image
    
    @staticmethod
    @numba.njit
    def bilinear(Q11, Q21, Q12, Q22, y_frac, x_frac):
        # if Q11.ndim == 3:
        #     y_frac = np.expand_dims(y_frac, 2)
        #     x_frac = np.expand_dims(x_frac, 2)
        new_image = (1 - y_frac) * (1 - x_frac) * Q11 + \
                    (1 - y_frac) * x_frac * Q12 + \
                    y_frac * (1 - x_frac) * Q21 + \
                    y_frac * x_frac * Q22
        return new_image



# if __name__ == '__main__':
#     import cv2
#     from matplotlib import pyplot as plt
#     import time

#     h, w = (400,500)
#     new_h, new_w = (300,400)
#     # x = np.arange(0, w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
#     y = np.arange(0, h).reshape(-1, 1, 1).repeat(w, axis=1).repeat(3, axis=2).astype(np.float32)
#     # plt.figure()
#     # plt.imshow(x)
#     # plt.figure()
#     # plt.imshow(y)
#     resize = Resize(h, w, new_h, new_w, 'bilinear')

#     # x1 = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     y1 = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     tic = time.perf_counter()
#     for _ in range(10):
#         y1 = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     print(f'cv2 resize: {time.perf_counter()-tic}s')

#     # x2 = resize(x, new_h, new_w, 'max')
#     y2 = resize.resize(y, new_h, new_w, 'bilinear')
#     tic = time.perf_counter()
#     for _ in range(10):
#         y2 = resize.resize(y, new_h, new_w, 'bilinear')
#     print(f'our resize: {time.perf_counter()-tic}s')
#     print(f'max abs error: {np.abs(y1-y2).max()}')
#     # plt.figure()
#     # plt.imshow(y1)
#     # plt.figure()
#     # plt.imshow(y2)
#     # plt.show(block=True)


if __name__ == '__main__':
    # import lz4.frame as lz
    # import pickle as plk
    # import os
    # import cv2
    # def read_lz4(lz4_file):
    #     if os.path.exists(lz4_file):
    #         with open(lz4_file, 'rb') as f:
    #             cdata = plk.load(f)
    #         arr = lz.decompress(cdata['arr'])
    #         arr = np.frombuffer(arr, dtype = cdata['dtype'])
    #         arr = np.reshape(arr, cdata['shape'])
    #         #print(arr)
    #         return arr   

    # disp_gt = read_lz4(r"C:\Users\dengy\Desktop\tmp\1702958788460187392.lz4")
    # disp_min, disp_max = disp_gt.min(), disp_gt.max()
    # new_h, new_w = (2740,1920)
    # resize = Resize()
    # disp_gt = resize.resize(disp_gt, new_h, new_w, 'max')
    # disp_min, disp_max = disp_gt.min(), disp_gt.max()
    # disp_gt_vis = (disp_gt - disp_min) / (disp_max - disp_min) * 255.0
    # disp_gt_vis = disp_gt_vis.astype("uint8")
    # disp_gt_vis = cv2.applyColorMap(disp_gt_vis, 12)
    # cv2.imwrite('disp_max_numpy.webp', disp_gt_vis)
    print('resize main')

