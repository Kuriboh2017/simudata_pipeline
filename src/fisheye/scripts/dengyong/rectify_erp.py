#!/usr/bin/env python3
import time
from enum import Enum
import math
import numpy as np
import cv2


class RectiSpec:
    def __init__(self, row, col, param, axis):
        self.row = row
        self.col = col
        self.param = param
        self.axis = axis


def generate_remap_table(left_intri, right_intri, translation, rotation, recti_spec):
    irmat = rotation.transpose()
    itrans = -1 * np.dot(irmat, translation.transpose())
    rmat1 = np.eye(3, 3)
    e1 = itrans.transpose() / np.linalg.norm(itrans)
    e2 = np.matrix([-itrans[1], itrans[0], 0]).reshape(1, 3)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    e3 = e3 / np.linalg.norm(e3)
    rmat1[0, :] = e1
    rmat1[1, :] = e2
    rmat1[2, :] = e3
    rmat2 = np.dot(rmat1, irmat)

    left_remap_x_list = []
    left_remap_y_list = []
    right_remap_x_list = []
    right_remap_y_list = []

    left_x, left_y = remap_camera(
        left_intri, recti_spec.param, rmat1, recti_spec.row, recti_spec.col)
    right_x, right_y = remap_camera(
        right_intri, recti_spec.param, rmat2, recti_spec.row, recti_spec.col)
    left_remap_x_list.append(left_x)
    left_remap_y_list.append(left_y)
    right_remap_x_list.append(right_x)
    right_remap_y_list.append(right_y)

    # concatenate along y axis
    left_remap_x = np.concatenate(left_remap_x_list, axis=0)
    left_remap_y = np.concatenate(left_remap_y_list, axis=0)
    right_remap_x = np.concatenate(right_remap_x_list, axis=0)
    right_remap_y = np.concatenate(right_remap_y_list, axis=0)
    return left_remap_x, left_remap_y, right_remap_x, right_remap_y


def remap_camera(calib_param, recti_param, rmat, recti_row, recti_col):
    irot = rmat.T
    mapx = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    mapy = np.zeros((int(recti_row), int(recti_col)), dtype=np.float32)
    for v in range(int(recti_row)):
        for u in range(int(recti_col)):
            xn = (u - recti_param[2]) / recti_param[0]
            yn = (v - recti_param[3]) / recti_param[1]
            p3d = np.array([xn, yn, 1.])
            p3d[0] = np.sin(xn)
            p3d[1] = np.cos(xn) * np.sin(yn)
            p3d[2] = np.cos(xn) * np.cos(yn)
            p3d_new = np.dot(irot, p3d)

            x = p3d_new[0]
            y = p3d_new[1]
            z = p3d_new[2]

            fx, fy, cx, cy = calib_param[:4]
            alpha, beta = calib_param[4], calib_param[5]
            r2 = x * x + y * y
            rho2 = beta * r2 + z * z
            rho = np.sqrt(rho2)
            norm = alpha * rho + (1. - alpha) * z
            mx, my = x / norm, y / norm

            mapx[v, u] = fx * mx + cx
            mapy[v, u] = fy * my + cy
    return mapx, mapy


def read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
    right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6

    translation = fs.getNode("trans").mat().reshape(-1) * 0.01
    rotation = fs.getNode("rmat").materp_img_down.png  erp_img_down_truth.png  erp_img_down_updated.png

    return left_intri, right_intri, translation, rotation


if __name__ == "__main__":
    # test read yaml
    l, r, trans, rmat = read_yaml(r"rectification.yml")
    # trans = np.array([-1,0,0])
    print(l, r, trans, rmat)

    # test generate table
    start = time.time()
    recti_spec=RectiSpec(1408,1280,[611.1549814728781,436.0673381319995,639.5,703.5],0) # row, col, param, axis

    lx, ly, rx, ry = generate_remap_table(l, r, trans, rmat, recti_spec)
    print(time.time() - start)
    # print(lx, ly, rx, ry)
    np.savez_compressed('lut_fisheye2erp_updated.npz', lx=lx, ly=ly, rx=rx, ry=ry)
    # np.savez_compressed('lut_sensors_data_2023.08.14-15.36.45-2023.08.14-15.37.16.npz', lx=lx, ly=ly, rx=rx, ry=ry)

    # # test image
    # test_img = cv2.imread(r"D:\Codes\rectify_tools\sensors_data_2023.08.14-15.36.45-2023.08.14-15.37.16\group1\cam1_0\src_img\217.121290535.png")
    # # cv2.imshow("ori", test_img)
    # test_img_remaped = cv2.remap(test_img, lx, ly, cv2.INTER_LINEAR)
    # # cv2.imshow("remaped", test_img_remaped)
    # cv2.imwrite(r"D:\Codes\rectify_tools\sensors_data_2023.08.14-15.36.45-2023.08.14-15.37.16\group1_ERP\cam1_0\Image\217.121290535_new.png", test_img_remaped)

    # # test image
    # test_img = cv2.imread(r"D:\Codes\rectify_tools\sensors_data_2023.08.14-15.36.45-2023.08.14-15.37.16\group1\cam1_1\src_img\217.121290535.png")
    # # cv2.imshow("ori", test_img)
    # test_img_remaped = cv2.remap(test_img, rx, ry, cv2.INTER_LINEAR)
    # # cv2.imshow("remaped", test_img_remaped)
    # cv2.imwrite(r"D:\Codes\rectify_tools\sensors_data_2023.08.14-15.36.45-2023.08.14-15.37.16\group1_ERP\cam1_1\Image\217.121290535_new.png", test_img_remaped)

