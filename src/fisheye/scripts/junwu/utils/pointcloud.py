import numpy as np

def disp2pointcloud_pinhole(disp, fx, fy, cx, cy, baseline, max_depth=1.0e6, gt_depth=None):
    baseline = abs(baseline)
    H, W = disp.shape
    grid = np.mgrid[0:H, 0:W]
    v, u = grid[0], grid[1] # u = [0, W-1], v = [0, H-1]

    if gt_depth is None:
        depth = baseline * fx / disp.clip(min=1.0e-3)
    else:
        depth = gt_depth
    depth[depth>max_depth] = max_depth
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_cloud = np.stack([x,y,z], axis=-1)

    return point_cloud

def disp2pointcloud_fisheye(disp, fx, fy, cx, cy, baseline, max_depth, rmat=None):
    baseline = abs(baseline)
    H, W = disp.shape
    grid = np.mgrid[0:H, 0:W]
    v, u = grid[0], grid[1]  # u = [0, W-1], v = [0, H-1]

    depth = baseline * fx / disp.clip(min=1.0e-3)
    depth[depth > max_depth] = max_depth
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_cloud = np.stack([x, y, z], axis=-1)

    if rmat is not None:
        shape = point_cloud.shape
        point_cloud = point_cloud.reshape(-1, 3)
        point_cloud = np.transpose(point_cloud)
        point_cloud = np.transpose(np.dot(rmat, point_cloud))  # =3xN =>Nx3
        point_cloud = point_cloud.reshape(shape)

    return point_cloud, depth