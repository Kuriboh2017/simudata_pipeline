import math

def calculate_focal_length(resolution, fov):
    return resolution / (2.0 * math.tan(fov / 2.0))

if __name__ == '__main__':
    resolution = 800
    fov = 65.65708358282505
    fov_rad = math.radians(fov)
    focal_length = calculate_focal_length(resolution, fov_rad)
    print(f'focal_length = {focal_length}') # 620.0

    resolution = 976
    fov = 68
    fov_rad = math.radians(fov)
    focal_length = calculate_focal_length(resolution, fov_rad)
    print(f'focal_length = {focal_length}') # 723.4897526342171
