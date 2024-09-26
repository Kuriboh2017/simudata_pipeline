#!/usr/bin/env python3
'''
TODO
'''
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import run
import argparse
import distutils.spawn
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


kOutputDir = 'output'
kIntermediateDir = 'intermediate'


def _append_filename(filename, appendix):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, appendix)


def _panorama_to_3pinholes(input_file, output_file, args):
    '''
    Input is a single image panorama filepath. Output is two round fisheye image filepaths.
    '''
    assert (isinstance(input_file, Path))
    folder_name = 'data_segmentation_gray' if args.segmentation_graymap else 'data_depth'
    output_file = output_file.replace('/data/', f'/{folder_name}/')
    output_dir = Path(output_file).parent
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_3pinholes_exe = distutils.spawn.find_executable(
        "panorama_to_3pinholes.py")
    assert panorama_to_3pinholes_exe, 'Error: executable `panorama_to_3pinholes.py` is not available!'
    cmds = [panorama_to_3pinholes_exe, f'--input-image-path={input_file}']
    output_filepath0 = output_dir / _append_filename(output_file, 'down')
    output_filepath1 = output_dir / _append_filename(output_file, 'up')
    cmds.append(
        f'--output-image-path0={str(output_filepath0)}')
    cmds.append(
        f'--output-image-path1={str(output_filepath1)}')
    cmds.append(
        f'--overwrite-remapping-filepath={str(args.panorama_to_3pinholes)}')
    cmds.append('--delete-odd-rows')
    if args.depth:
        cmds.append('--depth')
    if args.segmentation_graymap:
        cmds.append('--segmentation-graymap')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return [output_filepath0, output_filepath1]


def _process_image(input_output_pair, args):
    input_file, output_file = input_output_pair
    rectified_files = _panorama_to_3pinholes(
        Path(input_file), output_file, args)
    for rectified_file in rectified_files:
        if rectified_file.is_file():
            _logger.info(
                f'Generated a rectified file : {rectified_file}')


def _main(args):
    filelist = np.load(args.input_filelist, allow_pickle=True)
    key_name = 'depth_file_mapping' if args.depth else 'seg_gray_mapping'
    loaded_dict = filelist[key_name].item()
    input_output_pairs = list(loaded_dict.items())
    with Pool(cpu_count()) as p:
        p.map(partial(_process_image, args=args), input_output_pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run fisheye pipelines')
    parser.add_argument('-i', '--input-filelist', required=True,
                        help='filelist of the input data')
    parser.add_argument('--panorama-to-3pinholes', required=True,
                        help='remapping table filepath')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='whether or not processing depth image')
    parser.add_argument('--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run')
    args = parser.parse_args()
    _main(args)
