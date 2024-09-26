import os
import re
import numpy as np

def _get_folder_name(path):
    with open(path, 'r') as file:
        text = file.read()
    matches = re.findall(r'\b[\w_]+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', text)
    folder_names = np.unique(matches)
    print(f'Find {len(folder_names)} unique matched folder names')
    return folder_names

new_folder_names = _get_folder_name('synt_new_scene_syn_train_lz4y.csv')
old_folder_names = _get_folder_name('train/synt_0617_new_scene_syn_train_lz4y.csv')
extra_folder_names = _get_folder_name('/mnt/112-data/R23024/data1/junwu/tea_gi/data_transform/full_resolution_to_process.txt')
fog_folder_names = _get_folder_name('/mnt/112-data/R23024/data1/junwu/data/badcase/20230718_fog_glare.csv')

water_folder_names = _get_folder_name('/mnt/119-data/R10198/yanjianhua/test/20230718_fog_glare_sytn.csv')

set(new_folder_names) - set(old_folder_names)

def find_folder(root_path, folder_name):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if folder_name in dirnames:
            return os.path.join(dirpath, folder_name)
    return None


root_folder = '/mnt/112-data/R23024/data/junwu/data/'
folder_paths = []
found_folders = set()
for folder_name in folder_names:
    if folder_name in found_folders:
        continue
    folder_path = find_folder(root_folder, folder_name)
    if folder_path is not None:
        folder_paths.append(folder_path)
        found_folders.add(folder_name)


def _get_folder_name(path):
    with open(path, 'r') as file:
        text = file.read()
    matches = re.findall(r'\b[\w_]+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_out_renamed', text)
    folder_names = np.unique(matches)
    print(f'Find {len(folder_names)} unique matched folder names')
    return folder_names

folder_names = _get_folder_name('filelist_of_confusing_images.txt')
