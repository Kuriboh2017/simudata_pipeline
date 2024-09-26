import math
import numpy as np
import cv2
import itertools
import open3d as o3d

eucm_alpha = 0.652
eucm_beta = 1.077
fx, fy, cx, cy = 326.5, 326.5, 559.5, 559.5
depth_limit = 200


def _unproject_eucm(eucm_alpha, eucm_beta, x, y):
    r2 = x * x + y * y
    beta_r2 = eucm_beta * r2
    term_inside_sqrt = 1 - (2 * eucm_alpha - 1) * beta_r2
    if term_inside_sqrt < 0:
        return np.nan
    numerator = 1 - eucm_alpha * eucm_alpha * beta_r2
    denominator = eucm_alpha * math.sqrt(term_inside_sqrt) + 1 - eucm_alpha
    return numerator / denominator


def create_point_cloud(rgb_image_path, depth_image_path):
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = np.load(depth_image_path, allow_pickle=True)['arr_0']
    point_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for v, u in itertools.product(range(depth_image.shape[0]), range(depth_image.shape[1])):
        depth = depth_image[v, u]
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = _unproject_eucm(eucm_alpha, eucm_beta, x, y)
        if math.isnan(z) or depth == 0 or depth > depth_limit:
            continue
        point = np.array([x, y, z]) * depth
        points.append(point)
        color = rgb_image[v, u] / 255.0
        colors.append(color[::-1])
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


rgb_image_path = 'rgb_down.webp'
depth_image_path = 'depth_down.npz'

point_cloud = create_point_cloud(rgb_image_path, depth_image_path)

o3d.visualization.draw_geometries([point_cloud])

o3d.io.write_point_cloud("output.pcd", point_cloud, write_ascii=False)
point_cloud2 = o3d.io.read_point_cloud("output.pcd")
o3d.visualization.draw_geometries([point_cloud2])
