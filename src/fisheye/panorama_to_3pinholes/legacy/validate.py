import os
import cv2
import numpy as np
import argparse
import math
import multiprocessing as mlp
from functools import partial

up_npz_path = '/home/yuan/simulation/lens_effects/src/fisheye/validate/LUT_top_panorama1280.npz'
up_npz_file = np.load(up_npz_path)
# down_npz_file = np.load(down_npz_path)

up_npz_x_data = up_npz_file['mapx']
up_npz_y_data = up_npz_file['mapy']

l_img_path = '/home/yuan/tmp/a2_rot.png'
out_img_path = '/home/yuan/tmp/a2_truth.png'
l_img = cv2.imread(l_img_path)

l_up_remap = cv2.remap(l_img, up_npz_x_data.astype(
    np.float32), up_npz_y_data.astype(np.float32), interpolation=cv2.INTER_LINEAR)


cv2.imwrite(out_img_path, l_up_remap)

cv2.imshow("Remapped", l_up_remap)
cv2.waitKey(0)
cv2.destroyAllWindows()
