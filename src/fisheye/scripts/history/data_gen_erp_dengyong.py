import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


ext = '.png'
is_up = False



data = np.load(r'sample_with_new_intrinsics\rgb\remapping_table_panorama2fisheye_ERP.npz')
down_x, down_y, up_x, up_y = data['down_x'], data['down_y'], data['up_x'], data['up_y']

data = np.load('lut_fisheye2erp.npz')
lx, ly, rx, ry = data['lx'], data['ly'], data['rx'], data['ry']

# fisheye_img_path = r'sample_with_new_intrinsics\rgb\a_fisheye_0.png'
erp_img_path = r'eval_folder\Autel_Evening_Highway_2023-08-08-17-49-16_out\group1\cam1_1\Image'

file_list = os.listdir(erp_img_path)

def crop_erp(data, is_disp, is_up):
    crop_fov = (185, 120) # 训练dataloader输出的最终fov
    crop_shape = (1408, 1280)
    # resize
    resize_h = round(crop_shape[0] / crop_fov[0] * 360)
    resize_w = round(crop_shape[1] / crop_fov[1] * 180)
    dsize = (resize_w, resize_h)
    if is_disp:
        resize_scale = resize_w / data.shape[1]
        data = (cv2.resize(data.astype(np.float32), dsize=dsize, interpolation=cv2.INTER_NEAREST) * resize_scale).astype(np.float16) # (resize_h, resize_w, )
    else:
        data = cv2.resize(data, dsize=dsize, interpolation=cv2.INTER_LINEAR) # (resize_h, resize_w, )

    # w, center crop
    crop_w = crop_shape[1]
    rescale_w = data.shape[1]
    dy = int(round((rescale_w-crop_w)/2))+1
    data = data[:,dy:dy+crop_w] # (2740, 1920, 3) -> (2740, 1280, 3)

    if is_up:
        data = np.vstack([data[-19:, :], data[:1370+19, :]])
    else:
        data = np.vstack([data[1370-19:, :], data[:19, :]])
    # plt.imshow(data)
    # plt.show(block=True)
    return data

for erp_filename in file_list:

    full_erp_img = cv2.imread(os.path.join(erp_img_path, erp_filename))
    full_erp_disp = np.load(os.path.join(erp_img_path.replace('Image', 'Disparity'), erp_filename.replace('.webp', '.npz')))['data']

    fisheye_img_up = cv2.remap(full_erp_img, up_x, up_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imwrite("test_img_fisheye.png", fisheye_img_up)
    erp_img_up = cv2.remap(fisheye_img_up, lx, ly, cv2.INTER_LINEAR)
    erp_img_up = cv2.rotate(erp_img_up, cv2.ROTATE_180)
    os.makedirs(erp_img_path.replace('Image', 'Image_ERP'), exist_ok=True)
    cv2.imwrite(os.path.join(erp_img_path.replace('Image', 'Image_ERP'), erp_filename.replace('.webp', '_up.webp')), erp_img_up)

    fisheye_img_down = cv2.remap(full_erp_img, down_x, down_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imwrite("test_img_fisheye.png", fisheye_img_down)
    erp_img_down = cv2.remap(fisheye_img_down, lx, ly, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(erp_img_path.replace('Image', 'Image_ERP'), erp_filename.replace('.webp', '_down.webp')), erp_img_down)

    erp_img_up = crop_erp(full_erp_img, is_disp=False, is_up=True)
    os.makedirs(erp_img_path.replace('Image', 'Image_ERP_ori'), exist_ok=True)
    cv2.imwrite(os.path.join(erp_img_path.replace('Image', 'Image_ERP_ori'), erp_filename.replace('.webp', '_up.webp')), erp_img_up)

    erp_img_down = crop_erp(full_erp_img, is_disp=False, is_up=False)
    cv2.imwrite(os.path.join(erp_img_path.replace('Image', 'Image_ERP_ori'), erp_filename.replace('.webp', '_down.webp')), erp_img_down)

    erp_disp_up = crop_erp(full_erp_disp, is_disp=True, is_up=True)
    os.makedirs(erp_img_path.replace('Image', 'Disp_ERP'), exist_ok=True)
    np.savez_compressed(os.path.join(erp_img_path.replace('Image', 'Disp_ERP'), erp_filename.replace('.webp', '_up.npz')), data=erp_disp_up)

    erp_disp_down = crop_erp(full_erp_disp, is_disp=True, is_up=False)
    np.savez_compressed(os.path.join(erp_img_path.replace('Image', 'Disp_ERP'), erp_filename.replace('.webp', '_down.npz')), data=erp_disp_down)

