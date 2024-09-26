#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import re
import cv2
import itertools
import math
import numpy as np
import open3d as o3d
import os

kCurrentDir = os.path.dirname(os.path.abspath(__file__))


def _convert_panorama_uv_to_xyz(UV):
    angles = (2 * math.pi * (UV[0] + 0.5), math.pi * UV[1])
    val = math.sin(angles[1])
    return [val * math.sin(angles[0]), -val * math.cos(angles[0]), math.cos(angles[1])]


def _create_point_cloud(image_pair, output_dir, args):
    rgb_image_path, depth_image_path = image_pair
    print(f'Processing {rgb_image_path}')
    rgb_image = cv2.imread(str(rgb_image_path))
    depth_image = np.load(str(depth_image_path), allow_pickle=True)['arr_0']
    point_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for v, u in itertools.product(range(depth_image.shape[0]), range(depth_image.shape[1])):
        depth = depth_image[v, u]
        if depth <= 0 or depth > 200:
            continue
        un = (u + 0.) / depth_image.shape[1]
        vn = (v + 1.) / depth_image.shape[0]
        normal = _convert_panorama_uv_to_xyz([vn, un])
        point = np.array(normal) * depth
        points.append(point)
        color = rgb_image[v, u] / 255.0
        colors.append(color[::-1])
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    output_image_path = Path(output_dir) / \
        Path(rgb_image_path).with_suffix('.pcd').name
    o3d.io.write_point_cloud(str(output_image_path),
                             point_cloud, write_ascii=False, compressed=True)
    print(f'Point cloud outputted to {str(output_image_path)}')
    if args.visualize:
        o3d.visualization.draw_geometries([point_cloud])


def _find_matching_files(rgb_dir, depth_dir):
    rgb_files = {f.stem: f for f in Path(rgb_dir).iterdir() if f.is_file()}
    depth_files = {f.stem: f for f in Path(depth_dir).iterdir() if f.is_file()}
    matching_pairs = []
    for stem, rgb_file in rgb_files.items():
        depth_file = depth_files.get(stem)
        if depth_file:
            matching_pairs.append((rgb_file, depth_file))
    return matching_pairs


def _extract_integer_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


def _main(args):
    rgb_dir = Path(args.rgb_image_dir)
    depth_dir = Path(args.depth_image_dir)
    output_dir = Path(args.output_dir)
    assert rgb_dir.exists(
    ), f'Error: can\'t find the rgb image directory {rgb_dir}'
    assert depth_dir.exists(
    ), f'Error: can\'t find the depth image directory {depth_dir}'
    output_dir.mkdir(parents=True, exist_ok=True)
    every_n_frames = args.generate_every_n_frames
    assert every_n_frames > 0, 'Error: generate_every_n_frames must be greater than 0'

    matches = _find_matching_files(rgb_dir, depth_dir)
    sorted_matches = sorted(
        matches, key=lambda x: _extract_integer_from_filename(x[0].name))
    subset_matches = [pair for idx, pair in enumerate(
        sorted_matches) if idx % every_n_frames == 0]
    with Pool(cpu_count()) as p:
        p.map(partial(_create_point_cloud,
              output_dir=output_dir, args=args), subset_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate batches of point cloud files from RGB and depth images.')
    parser.add_argument('-rgb', '--rgb-image-dir', required=True,
                        help='input directory of the rgb images')
    parser.add_argument('-depth', '--depth-image-dir', required=True,
                        help='input directory of the depth images')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='output directory of point cloud files')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the point cloud')
    parser.add_argument('-d', '--depth-limit', type=int, default=200,
                        help='depth limit of the point cloud')
    parser.add_argument('-n', '--generate-every-n-frames', type=int, default=10,
                        help='generate point cloud every n images')
    args = parser.parse_args()
    _main(args)
