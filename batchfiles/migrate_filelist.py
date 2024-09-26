#!/usr/bin/env python3
from pathlib import Path
import random
import shutil
from datetime import datetime


new_root = '/mnt/112-data/R23024/data/junwu/data'


def assemble_paths(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    root_folder = [folder.strip() for folder in lines[0].split(',')][0]
    assembled_paths_2d = []
    new_path_root = Path(new_root)
    for line in lines[1:]:
        filenames = [filename.strip() for filename in line.split(',')]
        current_paths = [f"{root_folder}/{f}" for f in filenames]
        relative_paths = [str(Path(f).relative_to(new_path_root))
                          for f in current_paths]
        assembled_paths_2d.append(relative_paths)
    return assembled_paths_2d


def write_csv(paths, output_path):
    import csv
    with open(output_path, 'a') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in paths:
            if item:
                writer.writerow(item)


def backup_filelist(filelist):
    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    directory = filelist.parent / 'backup_filelist'
    directory.mkdir(parents=True, exist_ok=True)
    backup_filelist = directory / f'{filelist.stem}_{time_str}.csv'
    shutil.copyfile(filelist, backup_filelist)


def _main(args):
    src_filelist = Path(args.src_filelist)
    dst_train_filelist = Path(args.dst_train_filelist)
    dst_eval_filelist = Path(args.dst_eval_filelist)
    assert src_filelist.exists(
    ), f'Error: src_filelist `{src_filelist}` does not exist!'
    paths = assemble_paths(src_filelist)
    random.shuffle(paths)
    split_index = int(len(paths) * 0.97)
    train_paths = paths[:split_index]
    eval_paths = paths[split_index:]
    backup_filelist(dst_train_filelist)
    backup_filelist(dst_eval_filelist)
    write_csv(train_paths, dst_train_filelist)
    write_csv(eval_paths, dst_eval_filelist)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src-filelist',
                        help='The source filelist to be migrated.')
    parser.add_argument('-t', '--dst-train-filelist',
                        help='The destination train filelist.')
    parser.add_argument('-e', '--dst-eval-filelist',
                        help='The destination eval filelist.')
    args = parser.parse_args()
    _main(args)
