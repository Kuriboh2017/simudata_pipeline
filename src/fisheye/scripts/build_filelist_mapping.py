from pathlib import Path
import numpy as np


def _get_output_dir_id(number, images_per_folder=10):
    # 1 batch has |images_per_folder| images with the same remapping table.
    # 1 part has 100 batches.
    batches_per_part = 100
    images_per_part = images_per_folder * batches_per_part
    part_id = number // images_per_part
    batch_id = number % images_per_part // images_per_folder
    return part_id, batch_id


def _get_filelist_mapping(filelist, out_dir):
    rgb_left_file_mapping, depth_file_mapping, seg_gray_mapping = {}, {}, {}
    output_path = Path(out_dir)
    for id, line in enumerate(filelist):
        part_id, batch_id = _get_output_dir_id(id)
        output_dir = output_path / f'part{part_id}' / f'batch{batch_id}'
        p = Path(line)
        cam_name = p.parent.parent.name
        file_stem = p.stem
        #
        image_in = p.parent.parent / 'Image' / f'{file_stem}.webp'
        image_out = output_dir / f'{id}_{cam_name}_image.webp'
        #
        depth_in = p.parent.parent / 'Depth' / f'{file_stem}.npz'
        depth_out = output_dir / f'{id}_{cam_name}_disparity.npz'
        #
        seg_color_in = p.parent.parent / 'Segmentation' / \
            'Colormap' / f'{file_stem}.webp'
        seg_color_out = output_dir / f'{id}_{cam_name}_segmentation_color.webp'
        #
        seg_gray_in = p.parent.parent / 'Segmentation' / \
            'Graymap' / f'{file_stem}.webp'
        seg_gray_out = output_dir / f'{id}_{cam_name}_segmentation_gray.npz'
        #
        rgb_left_file_mapping[str(image_in)] = str(image_out)
        depth_file_mapping[str(depth_in)] = str(depth_out)
        rgb_left_file_mapping[str(seg_color_in)] = str(seg_color_out)
        seg_gray_mapping[str(seg_gray_in)] = str(seg_gray_out)
    return rgb_left_file_mapping, depth_file_mapping, seg_gray_mapping


def _get_right_filelist_mapping(filelist, out_dir):
    rgb_right_file_mapping = {}
    output_path = Path(out_dir)
    for id, line in enumerate(filelist):
        part_id, batch_id = _get_output_dir_id(id, 80)
        output_dir = output_path / f'part{part_id}' / f'batch{batch_id}'
        p = Path(line)
        fisheye_dir = p.parent.parent.parent
        file_stem = p.stem
        #
        cam1_name = 'cam1_1'
        image1_in = fisheye_dir / cam1_name / 'Image' / f'{file_stem}.webp'
        image1_out = output_dir / f'{id}_{cam1_name}_image.webp'
        rgb_right_file_mapping[str(image1_in)] = str(image1_out)
    return rgb_right_file_mapping


filelist = np.load('train_input_filelist.npz')['all_filepaths']
out_dir = '/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/data'
rgb_left_file_mapping, depth_file_mapping, seg_gray_mapping = _get_filelist_mapping(
    filelist, out_dir)

out_right_dir = '/mnt/115-rbd01/fisheye_dataset/train/data_noisy_by_file/data_right'
rgb_right_file_mapping = _get_right_filelist_mapping(filelist, out_right_dir)

np.savez_compressed('train_rgb_left_file_mapping.npz',
                    rgb_left_file_mapping=np.array(rgb_left_file_mapping))
np.savez_compressed('train_depth_file_mapping.npz',
                    depth_file_mapping=np.array(depth_file_mapping))
np.savez_compressed('train_seg_gray_mapping.npz',
                    seg_gray_mapping=np.array(seg_gray_mapping))
np.savez_compressed('train_rgb_right_file_mapping.npz',
                    rgb_right_file_mapping=np.array(rgb_right_file_mapping))
