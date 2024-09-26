import math
import numpy as np


def get_rotation(rpy_degrees):
    roll = math.radians(rpy_degrees[0])
    pitch = math.radians(rpy_degrees[1])
    yaw = math.radians(rpy_degrees[2])
    row1 = [math.cos(yaw) * math.cos(pitch),
            math.cos(yaw) * math.sin(pitch) * math.sin(roll) -
            math.sin(yaw) * math.cos(roll),
            math.cos(yaw) * math.sin(pitch) * math.cos(roll) +
            math.sin(yaw) * math.sin(roll)]
    row2 = [math.sin(yaw) * math.cos(pitch),
            math.sin(yaw) * math.sin(pitch) * math.sin(roll) +
            math.cos(yaw) * math.cos(roll),
            math.sin(yaw) * math.sin(pitch) * math.cos(roll) -
            math.cos(yaw) * math.sin(roll)]
    row3 = [-math.sin(pitch), math.cos(pitch) * math.sin(roll),
            math.cos(pitch) * math.cos(roll)]
    return np.array([row1, row2, row3])


def convert_xyz_to_panorama_uv(xyz):
    x, y, z = xyz
    u = 0.5 / math.pi * math.atan2(x, -y) + 0.5
    v = math.atan2(math.sqrt(x * x + y * y), z) / math.pi
    return np.array([u, v])


def unproject_eucm(eucm_alpha, eucm_beta, x, y):
    r2 = x * x + y * y
    beta_r2 = eucm_beta * r2
    term_inside_sqrt = 1 - (2 * eucm_alpha - 1) * beta_r2
    if term_inside_sqrt < 0:
        return np.nan
    numerator = 1 - eucm_alpha * eucm_alpha * beta_r2
    denominator = eucm_alpha * math.sqrt(term_inside_sqrt) + 1 - eucm_alpha
    return numerator / denominator
