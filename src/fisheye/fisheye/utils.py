import numpy as np
from scipy import interpolate


def _fill_nan_with_border_values(arr):
    non_nan_indices = np.argwhere(~np.isnan(arr))
    non_nan_values = arr[~np.isnan(arr)]
    nan_indices = np.argwhere(np.isnan(arr))
    filled_nan_values = interpolate.griddata(
        non_nan_indices, non_nan_values, nan_indices, method='nearest')
    arr[np.isnan(arr)] = filled_nan_values
    return arr


def _replace_boundary(arr):
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)
    arr[max_indices] = max_val - 1
    return arr


def customized_remap(input_img, col_data, row_data, func):
    col_data = _fill_nan_with_border_values(col_data.copy())
    row_data = _fill_nan_with_border_values(row_data.copy())
    col_data_lbound = np.floor(col_data).astype(int)
    col_data_ubound = np.ceil(col_data).astype(int)
    row_data_lbound = np.floor(row_data).astype(int)
    row_data_ubound = np.ceil(row_data).astype(int)

    # Handle out of bounds (e.g. image boundary 1280, 2560) indices
    col_data_lbound = _replace_boundary(col_data_lbound.copy())
    row_data_lbound = _replace_boundary(row_data_lbound.copy())
    col_data_ubound = _replace_boundary(col_data_ubound.copy())
    row_data_ubound = _replace_boundary(row_data_ubound.copy())
    depth_left_top = input_img[row_data_lbound, col_data_lbound]
    depth_left_bot = input_img[row_data_ubound, col_data_lbound]
    depth_right_top = input_img[row_data_lbound, col_data_ubound]
    depth_right_bot = input_img[row_data_ubound, col_data_ubound]
    depth_grid_in_channel = np.stack([
        depth_left_top, depth_left_bot, depth_right_top, depth_right_bot
    ])
    return func(depth_grid_in_channel, axis=0)


def _remap_segmentation_id(seg_graymap):
    seg_graymap = seg_graymap.copy()
    seg_graymap[seg_graymap == 1] = 0
    seg_graymap[seg_graymap == 2] = 0
    seg_graymap[seg_graymap == 0] = 1
    seg_graymap[seg_graymap == 11] = 0
    seg_graymap[seg_graymap == 18] = 2
    return seg_graymap


def _convert_segmentation_to_graymap(segmentation):
    is_equal = np.all(np.equal(segmentation[:, :, 0], segmentation[:, :, 1]) & np.equal(
        segmentation[:, :, 0], segmentation[:, :, 2]))
    assert is_equal, 'segmentation source image is not graymap'
    return segmentation[:, :, 0]
