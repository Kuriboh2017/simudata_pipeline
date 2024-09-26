import numpy as np
import cv2


def generate_array(length, start_id=0, reverse=False):
    ascending = list(range(256))
    descending = list(range(254, -1, -1))
    pattern = ascending + descending
    array = pattern
    while len(array) < 2*length:
        array += pattern[1:]
    if reverse:
        return array[-start_id:] + array[:length-start_id]
    return array[start_id:start_id+length]


def generate_gradient_image(width, height):
    data = [generate_array(width, y) for y in range(height)]
    reverse = [generate_array(width, y, True) for y in range(height)]
    img = np.zeros((height, width, 3), dtype=np.int16)
    for x in range(width):
        for y in range(height):
            img[y][x][0] = data[y][x]
            img[y][x][1] = reverse[y][x]
            img[y][x][2] = 0
    return img


def generate_debug_image_uint8(h, w):
    grid = np.mgrid[:h, :w]
    v, u = grid[0], grid[1]
    print('u[0,:10]', u[0, :10])
    xyz = np.zeros((h, w, 3), dtype=np.uint8)
    xyz[:, :, 0] = u % 256
    xyz[:, :, 1] = v % 256
    xyz[:, :, 2] = 128
    return xyz


def generate_debug_image_uint16(h, w):
    grid = np.mgrid[:h, :w]
    v, u = grid[0], grid[1]
    print('u[0,:10]', u[0, :10])
    xyz = np.zeros((h, w, 3), dtype=np.uint16)
    xyz[:, :, 0] = u
    xyz[:, :, 1] = v
    xyz[:, :, 2] = 128
    return xyz

# # Generate a gradient image from white in the center to black at the corners
# image = generate_gradient_image(1120, 1120)
# cv2.imwrite('gradient_int16.png', image)


debug_image = generate_gradient_image(1280, 2560)
cv2.imwrite('panorama_gradient.png', debug_image)
debug_image_bottom_half = debug_image[1280:, :]
cv2.imwrite('panorama_gradient_bottom_half.png', debug_image_bottom_half)


debug_image = generate_debug_image_uint8(1120, 1120)
cv2.imwrite('fisheye.png', debug_image)

debug_image = generate_debug_image_uint8(1362, 1280)
cv2.imwrite('3pinholes.png', debug_image)

debug_image = generate_debug_image_uint16(1362, 1280)
np.savez_compressed('3pinholes.npz', arr_0=debug_image)

debug_image = generate_debug_image_uint8(2560, 1280)
cv2.imwrite('panorama.png', debug_image)

debug_image_bottom_half = debug_image[1280:, :]
cv2.imwrite('panorama_bottom.png', debug_image_bottom_half)
