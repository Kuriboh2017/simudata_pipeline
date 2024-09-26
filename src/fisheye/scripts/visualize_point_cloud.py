#!/usr/bin/env python3
import argparse
from pathlib import Path
import open3d as o3d

def _main(args):
    file = args.input_file
    assert Path(file).exists(), f'Error: can\'t find the input file {file}'
    point_cloud2 = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([point_cloud2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True)
    args = parser.parse_args()
    _main(args)

