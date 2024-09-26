#!/usr/bin/env python3
import os
import math
import random
import time
from collections import namedtuple
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob

RectiSpec = namedtuple("RectiSpec", "row col param axis")

class Rectify:
    def __init__(self, fn):        
        fn_job = Path(fn).with_suffix(".joblib")
        if fn_job.exists():
            l, r, trans, rmat, multispecs = joblib.load(fn_job)
            multispecs = [RectiSpec(**a) for a in multispecs]
        else:
            l, r, trans, rmat, multispecs = self.read_yaml(fn)
            joblib.dump(
                [l, r, trans, rmat, [a._asdict() for a in multispecs]],
                fn_job)

        self.l, self.r, self.trans, self.rmat, self.multispecs = \
        l, r, trans, rmat, multispecs

    def read_yaml(self, path):
        path = str(path)
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        left_intri = fs.getNode("calibParam1").mat().reshape(-1)  # vec6
        right_intri = fs.getNode("calibParam2").mat().reshape(-1)  # vec6
        multi_recti_params = []
        multi_recti_params_node = fs.getNode("multiRectis")
        for i in range(multi_recti_params_node.size()):
            n = multi_recti_params_node.at(i)
            tmp = RectiSpec(n.at(3).real(), n.at(2).real(), [n.at(4).real(), n.at(
                5).real(), n.at(6).real(), n.at(7).real()], n.at(8).real())
            multi_recti_params.append(tmp)

        translation = fs.getNode("trans").mat().reshape(-1) * 0.01
        rotation = fs.getNode("rmat").mat()
        # for recification yml without multiRectis node
        if not multi_recti_params:
            """
            默认拆成1362(即454 * 3)的图
            """
            print('should not be here')
            row = 454
            col = 1280
            param = [368.92, 368.92, 639.5, 226.5]
            multi_recti_params = [
                RectiSpec(row=row, col=col, param=param, axis=-63.0),
                RectiSpec(row=row, col=col, param=param, axis=0.0),
                RectiSpec(row=row, col=col, param=param, axis=63.0)]

        return left_intri, right_intri, translation, rotation, multi_recti_params

    def random_aug(self, rmat, x=None):
        # print('x',x)
        rotation = R.from_matrix(rmat)
        euler_angle = rotation.as_euler('xyz', degrees=True)
        # if x is None:
        #     x = (np.random.rand(3) - 0.5) / 2
        # # x = [2, 2, 2]
        euler_angle += x
        rmat_new = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()

        return rmat_new

    def random_angle(self, mu=0, sigma=0.3, max=3):
        # mu, sigma = 0, 0.3  # mean and standard deviation
        ag_x = np.fmod(np.random.normal(mu, sigma) - mu, max) + mu
        ag_y = np.fmod(np.random.normal(mu, sigma) - mu, max) + mu
        ag_z = np.fmod(np.random.normal(mu, sigma) - mu, max) + mu
        angle_xyz = [ag_x, ag_y, ag_z]

        return angle_xyz

    def generate_remap_table(self, left_intri, translation, rotation, recti_specs, is_right=False,
                             angle=None, split=-1, aug_intri=False
                             ):
        irmat = rotation.transpose()
        itrans = -1 * np.dot(irmat, translation.transpose())  # include translation
        rmat1 = np.eye(3, 3)
        e1 = itrans.transpose() / np.linalg.norm(itrans)
        e2 = np.matrix([-itrans[1], itrans[0], 0]).reshape(1, 3)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        e3 = e3 / np.linalg.norm(e3)
        rmat1[0, :] = e1
        rmat1[1, :] = e2
        rmat1[2, :] = e3
        # print('rmat1 norm',np.linalg.norm(rmat1)/np.linalg.norm(np.eye(3, 3)),np.linalg.norm(e1),np.linalg.norm(e2),np.linalg.norm(e3))
        # print('irmat\n',irmat)
        # print('translation\n',translation)

        if is_right:  # include translation one more time??? Benny
            rmat1 = np.dot(rmat1, irmat)
            # rmat1 = np.dot(rotation, rmat1)

        if angle is not None:
            rmat_noise = self.random_aug(np.eye(3, 3), x=angle)
            rmat1 = np.dot(rmat1, rmat_noise)

        left_remap_x_list = []
        left_remap_y_list = []

        fx, fy, cx, cy, alpha, beta = left_intri[:6]  # fishey intri param
        # # add some noise to fx fy cx cy alpha beta
        if aug_intri:
            mu, sigma = 0, 2  # mean and standard deviation
            fx = fx + np.fmod(np.random.normal(mu, sigma), 10)
            fy = fy + np.fmod(np.random.normal(mu, sigma), 10)
            cx = cx + np.fmod(np.random.normal(mu, sigma), 10)
            cy = cy + np.fmod(np.random.normal(mu, sigma), 10)

            if random.random() < 0.5:
                mu2, sigma2 = 0, 0.01  # mean and standard deviation
                alpha = alpha + np.fmod(np.random.normal(mu2, sigma2), 0.2)
                beta = beta + np.fmod(np.random.normal(mu2, sigma2), 0.2)

        left_intri_new = [fx, fy, cx, cy, alpha, beta]

        for _, rectispec in enumerate(recti_specs):
            # print('math.cos(math.radians(rectispec.axis))',math.cos(math.radians(rectispec.axis)))
            rmat0 = np.array([[1, 0, 0],
                              [0, math.cos(math.radians(rectispec.axis)), -math.sin(math.radians(rectispec.axis))],
                              [0, math.sin(math.radians(rectispec.axis)), math.cos(math.radians(rectispec.axis))]])
            rmat1_rot = np.dot(rmat0, rmat1)
            # print('rmat1_rot\n',rmat1_rot)
            # print('matmul\n',np.matmul(rmat0, rmat1))
            # print('d\n',np.matmul(rmat0, rmat1)-rmat1_rot)

            # rotation = R.from_matrix(rmat1)
            # euler_angle = rotation.as_euler('xyz', degrees=True)
            # print('added euler_angle',euler_angle)
            # rotation = R.from_matrix(rmat1_rot)
            # euler_angle = rotation.as_euler('xyz', degrees=True)
            # print('euler_angle',euler_angle)

            left_x, left_y = self.remap_camera(
                left_intri_new, rectispec.param, rmat1_rot, rectispec.row, rectispec.col,
                split=split)

            left_remap_x_list.append(left_x)
            left_remap_y_list.append(left_y)

        # concatenate along y axis
        left_remap_x = np.concatenate(left_remap_x_list, axis=0)
        left_remap_y = np.concatenate(left_remap_y_list, axis=0)

        return left_remap_x, left_remap_y

    def generate_remap_table_2(self, left_intri,
                               angle=None, split=-1,
                               row=1120, col=1120,
                               translation=np.array([-1, 0, 0]),
                               ):
        itrans = -1 * np.dot(np.eye(3, 3), translation.transpose())  # include translation
        rmat1 = np.eye(3, 3)
        e1 = itrans.transpose() / np.linalg.norm(itrans)
        e2 = np.matrix([-itrans[1], itrans[0], 0]).reshape(1, 3)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        e3 = e3 / np.linalg.norm(e3)
        rmat1[0, :] = e1
        rmat1[1, :] = e2
        rmat1[2, :] = e3

        if angle is not None:
            rmat_noise = self.random_aug(np.eye(3, 3), x=angle)
            rmat1 = np.dot(rmat1, rmat_noise)

        left_x, left_y = self.remap_camera_2(
            left_intri, left_intri, rmat1, row, col,
            split=split)

        return left_x, left_y

    def remap_camera(self, calib_param, recti_param, rmat, recti_row, recti_col, split=-1):
        irot = rmat.T
        mapx, mapy = np.meshgrid(np.arange(int(recti_col)),
                                 np.arange(int(recti_row)))
        tmpx = (mapx - recti_param[2]) / recti_param[0]
        tmpy = (mapy - recti_param[3]) / recti_param[1]
        if 0 == split or 1 == split:  # split 0 or 1 to half sample the full image. Benny
            tmpx = tmpx[split::2]
            tmpy = tmpy[split::2]

        ones = np.ones_like(tmpx, dtype=tmpx.dtype)
        new = np.stack([tmpx, tmpy, ones], axis=-1)
        new = new.transpose(0, 2, 1)
        new = irot.dot(new)  # here rotate the mapx mapy (rotate 1to3,not fisheye), but want to rotate fisheye image
        new = new.transpose(1, 2, 0)
        rho = (new[..., :2] * new[..., :2]).sum(-1)
        fx, fy, cx, cy, alpha, beta = calib_param[:6]  # fishey intri param
        rho2 = beta * rho + new[..., -1] * new[..., -1]
        norm = alpha * np.sqrt(rho2) + (1 - alpha) * new[..., -1]
        mx = new[..., 0] / norm
        my = new[..., 1] / norm
        mx = fx * mx + cx
        my = fy * my + cy
        return mx.astype(np.float32), my.astype(np.float32)

    def remap_camera_2(self, calib_param, recti_param, rmat, recti_row, recti_col, split=-1):
        # print('recti_param',recti_param)
        # print('rmat\n',rmat)
        irot = rmat.T
        mapx, mapy = np.meshgrid(np.arange(int(recti_col)),
                                 np.arange(int(recti_row)))
        # print('mapx[:3,:3]\n',mapx[:3,:3])
        tmpx = (mapx - recti_param[2]) / recti_param[0]
        tmpy = (mapy - recti_param[3]) / recti_param[1]
        # print('tmpx[:3,:3]\n',tmpx[:3,:3])
        # if 0 == split or 1 == split: #split 0 or 1 to half sample the full image. Benny
        #     tmpx = tmpx[split::2]
        #     tmpy = tmpy[split::2]

        ones = np.ones_like(tmpx, dtype=tmpx.dtype)
        new = np.stack([tmpx, tmpy, ones], axis=-1)
        # print('new[0,0,:]',new[0,0,:])
        new = new.transpose(0, 2, 1)
        new = irot.dot(new)  # here rotate the mapx mapy (rotate 1to3,not fisheye), but want to rotate fisheye image
        new = new.transpose(1, 2, 0)
        # print('new[0,0,:]',new[0,0,:])
        # print('new.shape',new.shape)

        mx = new[..., 0] / new[..., 2]
        my = new[..., 1] / new[..., 2]

        mx = recti_param[0] * mx + recti_param[2]
        my = recti_param[1] * my + recti_param[3]

        # print('mx[:3,:3]\n',mx[:3,:3])

        return mx.astype(np.float32), my.astype(np.float32)

    def __call__(self, img, is_right=True, aug=False, angle=None, split=-1, aug_intri=False):
        # print('angle',angle)
        if isinstance(img, str):
            img = cv2.imread(img)
        l = self.r if is_right else self.l
        # if aug:
        #     rmat = self.random_aug(self.rmat,x=angle)
        # else:
        #     rmat = self.rmat

        lx, ly = self.generate_remap_table(l, self.trans, self.rmat, self.multispecs, is_right,
                                           angle=angle,
                                           split=split, aug_intri=aug_intri)
        test_img_remaped = cv2.remap(img, lx, ly, cv2.INTER_LINEAR)
        return test_img_remaped


def rectify_aug_4right(img_pkl_path, debug=False,split=0):
    # print('img_pkl_path', img_pkl_path)
    src_img_root = "/mnt/103-data/DataSets/Datasets123_pkl_1362x1280/SrcImagesFish"
    relative_path = "Autonomy" + str(img_pkl_path).rsplit("Autonomy", 1)[1]
    relative_path = relative_path.replace("_0/Image", "_1/src_img").replace(".pkl", ".png")
    yml_path = relative_path.rsplit("group", 1)
    yml_path = yml_path[0] + "group" + yml_path[1][0]
    yml_path = Path(src_img_root) / yml_path
    right_img_path = Path(src_img_root) / relative_path
    right_img_new = None
    
    yml_path = glob.glob(os.path.join(yml_path,"*_high.yml"))
    if len(yml_path)<1:
        return None
    else:
        pass
    
    if 1: #try:
        yml_path = yml_path[0]
        if 0:
            # temporary replace to a debug path
            yml_path = r'E:\rectify\y0\rectification_3in1_1684640399.yml'
            # yml_path = r'E:\rectify\python\BottomBinoRectification_high.yml'
            right_img_path = r'E:\rectify\y0\png\ROBOT_EVO3_8450_CALIB_STEREO_CAMERA_TOP_000007.png'
            print('yml_path', yml_path, 'right_img_path', right_img_path)
        rec_aug = Rectify(yml_path)
        if right_img_path.exists(): #for i in range(100):
            right_img = cv2.imread(str(right_img_path)) #[:, :1120]
            if right_img.shape != (1120, 1120, 3):
                return None

            angle = rec_aug.random_angle()
            # angle = [2, 0, 0]
            # angle =  None
            righ_flag = True
            split = split
            aug_intri = True
            if random.random() < 0.2:
                # rotate pinhole
                right_img_new = rec_aug(right_img, is_right=righ_flag, angle=angle, split=split, aug_intri=aug_intri)
            else:
                # rotate fisheye
                l = rec_aug.r if righ_flag else rec_aug.l
                # print('l', l)
                lx, ly = rec_aug.generate_remap_table_2(l, angle=angle)
                fisheye_2 = cv2.remap(right_img, lx, ly, cv2.INTER_LINEAR)
                right_img_new = rec_aug(fisheye_2, is_right=righ_flag, angle=None, split=split, aug_intri=aug_intri)
            if debug:
                i = 0
                o_dir = r'tmp/'
                o_dir = os.path.join(o_dir, str(i))
                os.makedirs(o_dir, exist_ok=True)
                if '\\' in str(right_img_path):
                    p = str(right_img_path).split('\\')[-1]
                else:
                    p = str(right_img_path).split('/')[-1]

                p1 = os.path.join(o_dir, p)
                img_ori = rec_aug(right_img, is_right=righ_flag, angle=None, split=split, aug_intri=False)
                print('p1', p1)
                cv2.imwrite(p1, img_ori)
                cv2.imwrite(p1.replace('.png', '_aug.png'), right_img_new)
                cv2.imwrite(p1.replace('.png', '_d.png'), (right_img_new - img_ori).astype(np.uint8))
                cv2.imwrite(p1.replace('.png', '_fe.png'), right_img)

                l = rec_aug.r if righ_flag else rec_aug.l
                lx, ly = rec_aug.generate_remap_table_2(l, angle=angle)
                fisheye_2 = cv2.remap(right_img, lx, ly, cv2.INTER_LINEAR)
                # print('fisheye_2.shape',fisheye_2.shape,fisheye_2.dtype)
                cv2.imwrite(p1.replace('.png', '_fe2.png'), fisheye_2)
                right_img_new2 = rec_aug(fisheye_2, is_right=righ_flag, angle=None, split=split, aug_intri=aug_intri)
                cv2.imwrite(p1.replace('.png', '_aug2.png'), right_img_new2)
                exit(0)
        else:
            # right_img_new = None
            return None
    # except Exception as e:
    #     right_img_new = None
    return right_img_new


import math


# import numpy as np


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotate_image(right_img, K_mat,
                 # angle_A=0.2,pixel_x_A=3,pixel_y_A=3,
                 angle_B=(0.3, 0.3, 0.1), pixel_x_B=3, pixel_y_B=3,
                 left_path=None,
                 ):
    # angle, pixel_x,pixel_y = (0.3, 0.3, 0.1), 3,3
    angle, pixel_x, pixel_y = angle_B, pixel_x_B, pixel_y_B

    if 1:
        ag_0 = angle[0]
        ag_1 = angle[1]
        ag_2 = angle[2]
        px_x = pixel_x
        px_y = pixel_y
        ag = np.deg2rad([ag_0, ag_1, ag_2])

    K_mat_new = K_mat.copy()
    K_mat_new[0, 2] += px_x
    K_mat_new[1, 2] += px_y

    R_mat = eulerAnglesToRotationMatrix([x for x in ag])

    if 1:
        rmat_new = R.from_euler('xyz', [-x for x in ag], degrees=True).as_matrix()
        print('ag', ag)
        print('R_mat\n', R_mat)
        print('rmat_new\n', rmat_new)
        print('d', rmat_new - R_mat)

        R_mat = rmat_new

    H_mat = K_mat_new.dot(R_mat).dot(np.linalg.inv(K_mat))
    H_mat = H_mat / H_mat[2][2]

    right_img = cv2.warpPerspective(
        right_img, H_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    return right_img


def test():
    import os
    list_path = '/mnt/data/data1/zhenpeng/Data/modelx_hires/train_filelist/20230517/real_new_scene_syn_train_lz4y.csv'
    lrimg_lst = []
    ldisp_lst = []
    lrseg_lst = []

    with open(list_path) as f_lst:
        for line_idx, line in enumerate(f_lst):
            if line_idx == 0:
                data_roots = line.strip().split(',')
            else:
                data_paths = line.strip().split(',')
                lrimg_lst.append(data_paths[0])
                ldisp_lst.append(data_paths[1])
                lrseg_lst.append(data_paths[2])
    count = 0
    for i in range(len(lrimg_lst)):
        p = os.path.join(data_roots[0], lrimg_lst[i])
        if i % 10000 == 0:
            print('i', i, p)
        img = rectify_aug_4right(p)
        if img is not None:
            count +=1
            print('--------------', img.shape, i, count, count/i,p,)
            # break


if __name__ == "__main__":
    # test()
    # p = '/mnt/103-data/DataSets/Datasets123_pkl_1362x1280/SrcImagesFish/Autonomy04/20221020_out/sensors_data_2022.10.08-04.40.03_out/group1/cam1_1/src_img/244035826354.png'
    p = '/mnt/103-data/DataSets/Datasets123_pkl_1362x1280/SrcImagesFish/Autonomy04/20221020_out/sensors_data_2022.10.08-04.40.03_out/group0/cam0_0/Image/396264314614.pkl'
    img = rectify_aug_4right(p)
    if img is not None:
        print('--------------', img.shape, p)
    else:
        print("None")
    # right_img = cv2.imread(p)
    # print(right_img.shape)
