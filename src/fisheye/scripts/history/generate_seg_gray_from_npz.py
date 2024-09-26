#!/usr/bin/env python3
'''
Generate segmentation graymap images from panorama images.
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


def _panorama_to_3pinholes(input_file, output_files, args):
    '''
    Input is a single image panorama filepath. Output is two 3pinholes image filepaths.
    '''
    assert (isinstance(input_file, Path))
    output_filepath0, output_filepath1 = output_files
    output_dir = Path(output_filepath0).parent
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    panorama_to_3pinholes_exe = distutils.spawn.find_executable(
        "panorama_to_3pinholes.py")
    assert panorama_to_3pinholes_exe, 'Error: executable `panorama_to_3pinholes.py` is not available!'
    cmds = [
        panorama_to_3pinholes_exe,
        f'--input-image-path={input_file}',
        f'--output-image-path0={str(output_filepath0)}',
        f'--output-image-path1={str(output_filepath1)}',
        f'--overwrite-remapping-filepath={str(args.panorama_to_3pinholes)}',
        '--delete-odd-rows',
    ]
    if args.segmentation_graymap:
        cmds.append('--segmentation-graymap')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return [output_filepath0, output_filepath1]


def _process_image(input_output_pair, args):
    input_file, output_files = input_output_pair
    _panorama_to_3pinholes(Path(input_file), output_files, args)
    _logger.info(f'Generated rectified files for {input_file}')


def _main(args):
    filelist = np.load(args.input_filelist, allow_pickle=True)
    key_name = 'folder_src_to_dst'
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
    parser.add_argument('--segmentation-graymap', action='store_true',
                        help='whether or not processing segmentation graymap image')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run')
    args = parser.parse_args()
    _main(args)
