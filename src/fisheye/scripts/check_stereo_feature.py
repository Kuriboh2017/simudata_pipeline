#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt


def do_verify_calibration_by_orb(left, right, rectified_image_path_i):
    # 读取需要特征匹配的两张照片，格式为灰度图
    img1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # BFMatcher匹配
    orb = cv2.ORB_create()  # 建立orb特征检测器
    kp1, des1 = orb.detectAndCompute(img1, None)  # 计算img1中的特征点和描述子
    kp2, des2 = orb.detectAndCompute(img2, None)  # 计算img2中的
    # 无法拾取特征点则返回NaN
    if not kp1 or not kp2:
        mean_x_diff = mean_y_diff = np.nan
        return (mean_x_diff, mean_y_diff)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 建立匹配关系
    matches = bf.match(des1, des2)  # 匹配描述子
    matches = sorted(matches, key=lambda x: x.distance)  # 据距离来排序
    matched_point = np.zeros((len(matches[:40]), 4), dtype=float)
    for i in range(len(matches[:40])):
        matched_point[i][:2] = kp1[matches[i].queryIdx].pt
        matched_point[i][2:] = kp2[matches[i].trainIdx].pt
    # 求纵向偏移中位数
    x_diff = (matched_point[:, 0] - matched_point[:, 2])
    y_diff = abs(matched_point[:, 1] - matched_point[:, 3])
    matches_array = np.array(matches[:40])
    matches_array_false = matches_array[np.logical_or(x_diff < -1, y_diff > 1)]
    matches_array_true = matches_array[np.logical_and(
        x_diff >= -1, y_diff <= 1)]
    mean_x_diff = np.median(x_diff)
    mean_y_diff = np.median(y_diff)
    img_visual1 = cv2.drawMatches(img1=left,
                                  keypoints1=kp1,
                                  img2=right,
                                  keypoints2=kp2,
                                  matches1to2=matches_array_true.tolist(),
                                  outImg=None,
                                  matchColor=(0, 255, 0),
                                  flags=0)  # 画出匹配关系
    img_visual2 = cv2.drawMatches(img1=left,
                                  keypoints1=kp1,
                                  img2=right,
                                  keypoints2=kp2,
                                  matches1to2=matches_array_false.tolist(),
                                  outImg=None,
                                  matchColor=(0, 0, 255),
                                  flags=6)  # 画出匹配关系，flags为偶数
    # img_visual = cv2.add(img_visual2,img_visual1)
    img_visual = cv2.addWeighted(img_visual2, 0.6, img_visual1, 0.4, 0)
    plt.imshow(cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB))
    plt.savefig(rectified_image_path_i, dpi=500, bbox_inches='tight')
    return (mean_x_diff, mean_y_diff)


def _main(args):
    left = cv2.imread(args.left_image_path)
    right = cv2.imread(args.right_image_path)
    rectified_image_path = args.rectified_image_path
    do_verify_calibration_by_orb(left, right, rectified_image_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Verify calibration by ORB.')
    parser.add_argument('-l', '--left-image-path', required=True,
                        help='path of the left image')
    parser.add_argument('-r', '--right-image-path', required=True,
                        help='path of the right image')
    parser.add_argument('-o', '--rectified-image-path', default='stereo_checking.png',
                        help='path of the rectified image')
    args = parser.parse_args()
    _main(args)
