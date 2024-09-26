from fisheye.core.types import RemappingTableParams, RemappingTable
from fisheye.core.utils import get_rotation, unproject_eucm, convert_xyz_to_panorama_uv
import math
import numpy as np


def generate_panorama_to_fisheye_table(params: RemappingTableParams) -> RemappingTable:
    down_x, down_y = _generate_p2f_remap_table(params, True)
    up_x, up_y = _generate_p2f_remap_table(params, False)
    return RemappingTable(down_x, down_y, up_x, up_y)


def generate_panorama_to_3pinholes_table(params: RemappingTableParams) -> RemappingTable:
    down_x, down_y = _generate_p2p_remap_table(params, True)
    up_x, up_y = _generate_p2p_remap_table(params, False)
    return RemappingTable(down_x, down_y, up_x, up_y)


def generate_fisheye_to_3pinholes_table(params: RemappingTableParams) -> RemappingTable:
    down_x, down_y = _generate_f2p_remap_table(params)
    return RemappingTable(down_x, down_y, None, None)


def _generate_p2f_remap_table(params: RemappingTableParams, down: bool):
    dst = params.dst
    fx, fy, cx, cy, eucm_alpha, eucm_beta = dst.intrinsics
    pitch = -90 if down else 90
    rmat_up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                                [0, 1, 0],
                                [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])
    rpy_noise_rot = get_rotation(params.rpy)
    mapx = np.zeros((int(dst.height), int(dst.width)), dtype=np.float32)
    mapy = np.zeros((int(dst.height), int(dst.width)), dtype=np.float32)
    for v in range(int(dst.height)):
        for u in range(int(dst.width)):
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            zn = unproject_eucm(eucm_alpha, eucm_beta, xn, yn)
            if math.isnan(zn):
                mapx[v, u] = np.nan
                mapy[v, u] = np.nan
                continue
            normal = np.array([xn, yn, zn])
            normal = np.dot(rpy_noise_rot.T, normal)
            normal = np.dot(rmat_up_or_down.T, normal)
            uv = convert_xyz_to_panorama_uv(normal)
            mapx[v, u] = uv[0] * params.src.width-0.5
            mapy[v, u] = uv[1] * params.src.height
    return mapy, mapx


def _p2p_remap_camera(params: RemappingTableParams, rmat: np.ndarray):
    src, dst = params.src, params.dst
    dst_width, dst_height = dst.width, dst.height // 3
    fx, fy, cx, cy = dst.intrinsics
    irot = rmat.T
    mapx = np.zeros((int(dst_height), int(dst_width)), dtype=np.float32)
    mapy = np.zeros((int(dst_height), int(dst_width)), dtype=np.float32)
    for v in range(int(dst_height)):
        for u in range(int(dst_width)):
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            p3d = np.array([xn, yn, 1.])
            p3d_new = np.dot(irot, p3d)
            uv = convert_xyz_to_panorama_uv(p3d_new)
            mapx[v, u] = uv[0] * src.width-0.5
            mapy[v, u] = uv[1] * src.height
    return mapy, mapx


def _generate_p2p_remap_table(params: RemappingTableParams, down: bool):
    left_remap_x_list = []
    left_remap_y_list = []
    pitch = -90 if down else 90
    up_or_down = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                           [0, 1, 0],
                           [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])
    rpy_noise_rot = get_rotation(params.rpy)
    for roll in (-63, 0, 63):
        r_rad = math.radians(roll)
        pinhole63 = np.array([[1, 0, 0],
                              [0, math.cos(r_rad), -math.sin(r_rad)],
                              [0, math.sin(r_rad), math.cos(r_rad)]])
        rot = np.dot(pinhole63, np.dot(rpy_noise_rot, up_or_down))
        left_x, left_y = _p2p_remap_camera(params, rot)
        left_remap_x_list.append(left_x)
        left_remap_y_list.append(left_y)
    left_remap_x = np.concatenate(left_remap_x_list, axis=0)
    left_remap_y = np.concatenate(left_remap_y_list, axis=0)
    return left_remap_x, left_remap_y


def _generate_f2p_remap_table(params: RemappingTableParams):
    remap_x_list = []
    remap_y_list = []
    for roll in (-63, 0, 63):
        r_rad = math.radians(roll)
        mat = np.array([[1, 0, 0],
                        [0, math.cos(r_rad), -math.sin(r_rad)],
                        [0, math.sin(r_rad), math.cos(r_rad)]])
        x, y = _f2p_remap_camera(params, mat)
        remap_x_list.append(x)
        remap_y_list.append(y)
    remap_x = np.concatenate(remap_x_list, axis=0)
    remap_y = np.concatenate(remap_y_list, axis=0)
    return remap_x, remap_y


def _f2p_remap_camera(params: RemappingTableParams, rmat: np.ndarray):
    src, dst = params.src, params.dst
    src_fx, src_fy, src_cx, src_cy = src.intrinsics
    dst_fx, dst_fy, dst_cx, dst_cy, eucm_alpha, eucm_beta = dst.intrinsics
    dst_width, dst_height = dst.width, dst.height // 3
    irot = rmat.T
    mapx = np.zeros((int(dst_height), int(dst_width)), dtype=np.float32)
    mapy = np.zeros((int(dst_height), int(dst_width)), dtype=np.float32)
    for v in range(int(dst_height)):
        for u in range(int(dst_width)):
            xn = (u - dst_cx) / dst_fx
            yn = (v - dst_cy) / dst_fy
            p3d = np.array([xn, yn, 1.])
            p3d_new = np.dot(irot, p3d)
            x = p3d_new[0]
            y = p3d_new[1]
            z = p3d_new[2]
            r2 = x * x + y * y
            rho2 = eucm_beta * r2 + z * z
            rho = np.sqrt(rho2)
            norm = eucm_alpha * rho + (1. - eucm_alpha) * z
            mx, my = x / norm, y / norm
            mapx[v, u] = src_fx * mx + src_cx
            mapy[v, u] = src_fy * my + src_cy
    return mapx, mapy
