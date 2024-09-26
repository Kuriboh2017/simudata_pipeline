#!/usr/bin/env python3
import argparse
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import cv2
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


# Old function, not used anymore.
def get_pixels_of_interest(rgb_image, depth_image, rgb_thresh=30, depth_thresh=10):
    '''
    Find pixels where RGB is similar (below the threshold) and
    depth changes sharply (above the threshold).
    '''
    grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    depth_magnitude = cv2.magnitude(grad_x, grad_y)
    rgb_diff_x = rgb_image[:-1, :-1] - rgb_image[1:, :-1]
    rgb_diff_y = rgb_image[:-1, :-1] - rgb_image[:-1, 1:]
    rgb_magnitude = np.sqrt(
        np.sum(np.square(rgb_diff_x), axis=2) + np.sum(np.square(rgb_diff_y), axis=2))
    depth_binary = (depth_magnitude[:-1, :-1] > depth_thresh).astype(np.uint8)
    rgb_binary = (rgb_magnitude < rgb_thresh).astype(np.uint8)
    return cv2.bitwise_and(rgb_binary, depth_binary)


def get_pixels_of_interest_sobel(rgb_image, depth_image, rgb_thresh=30, depth_thresh=10):
    '''
    Find pixels where RGB is similar (below the threshold) and
    depth changes sharply (above the threshold).
    '''
    grad_x_depth = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_depth = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    depth_magnitude = cv2.magnitude(grad_x_depth, grad_y_depth)
    # RGB
    grad_x_r = cv2.Sobel(rgb_image[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_r = cv2.Sobel(rgb_image[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
    grad_x_g = cv2.Sobel(rgb_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_g = cv2.Sobel(rgb_image[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
    grad_x_b = cv2.Sobel(rgb_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3)
    grad_y_b = cv2.Sobel(rgb_image[:, :, 2], cv2.CV_64F, 0, 1, ksize=3)
    r_magnitude = cv2.magnitude(grad_x_r, grad_y_r)
    g_magnitude = cv2.magnitude(grad_x_g, grad_y_g)
    b_magnitude = cv2.magnitude(grad_x_b, grad_y_b)
    rgb_magnitude = np.sqrt(np.square(r_magnitude) +
                            np.square(g_magnitude) + np.square(b_magnitude))
    depth_binary = (depth_magnitude > depth_thresh).astype(np.uint8)
    rgb_binary = (rgb_magnitude < rgb_thresh).astype(np.uint8)
    return cv2.bitwise_and(rgb_binary, depth_binary)


def get_percentage_below_threshold(depth_img, threshold=30):
    '''
    Get the percentage of pixels below the threshold.
    '''
    _, binary_mask = cv2.threshold(
        depth_img, threshold, 255, cv2.THRESH_BINARY_INV)
    count_below_threshold = cv2.countNonZero(binary_mask)
    total_pixels = depth_img.shape[0] * depth_img.shape[1]
    return (count_below_threshold / total_pixels) * 100


def get_image_file_list(data_root):
    return [
        str(file_path)
        for file_path in Path(data_root).rglob('*')
        if file_path.is_file()
        and '.webp' in file_path.suffix
        and 'cam0_0' in str(file_path)
        and 'Image' in str(file_path)
    ]


def _process_one(src_limg_path, args):
    src_depth_path = src_limg_path.replace(
        'Image', 'Depth').replace('.webp', '.npz')
    rgb_image = cv2.imread(src_limg_path)
    depth_image = np.load(src_depth_path)['arr_0']
    result_image = get_pixels_of_interest_sobel(
        rgb_image, depth_image.astype(np.float32))
    total_pixels = result_image.shape[0] * result_image.shape[1]
    count = cv2.countNonZero(result_image)
    percentage_of_interest_pixels = count / total_pixels * 100
    _logger.info(f'{src_limg_path}: {percentage_of_interest_pixels:.2f}%')
    near_pixels_percentage = get_percentage_below_threshold(depth_image)
    # Near pixels (below 30 meter) should be more than 50% inside an image to be considered.
    if (near_pixels_percentage > 50 and
            percentage_of_interest_pixels > args.interested_pixels_percentage_threshold):
        if args.save_interested_pixels_image:
            p = Path(src_limg_path)
            interest_pixels_img = p.parent / \
                f'{p.stem}_interest_pixels{p.suffix}'
            cv2.imwrite(interest_pixels_img, result_image * 255)
        return src_limg_path, percentage_of_interest_pixels
    else:
        return None, None


def _main(args):
    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f'{input_dir} does not exist'
    images = get_image_file_list(input_dir)
    _logger.info(f'Found {len(images)} images in {input_dir}')

    with Pool(cpu_count()) as pool:
        results = pool.map(partial(_process_one, args=args), images)

    selected = [r for r in results if r[0] is not None]
    num_selected = len(selected)
    print('Selected rate (%d/%d): %.2f' %
          (num_selected, len(results), num_selected/len(results)))

    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    filepath = f'filelist_of_interested_left_img_path_{time_str}.txt'
    sorted_selected = sorted(selected, key=lambda x: x[1], reverse=True)
    with open(filepath, 'w') as f:
        for p, percent in sorted_selected:
            f.write(f'{str(Path(p).absolute())},{percent:.2f}%\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Find images with similar RGB and different depth. ')
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory of the input images')
    parser.add_argument('-s', '--save-interested-pixels-image', action='store_true',
                        help='Save the image of interested pixels')
    parser.add_argument('-t', '--interested-pixels-percentage-threshold', type=float, default=4,
                        help='Threshold of percentage of interested pixels')
    args = parser.parse_args()
    _main(args)
