
import os
from pathlib import Path


def count_files_in_dir(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


total = 0
not_found_folders = []
for folder in folder_paths:
    image_folder = Path(folder) / 'group1'/'cam1_0'/'Image'
    try:
        num_of_files = count_files_in_dir(image_folder)
    except FileNotFoundError:
        not_found_folders.append(image_folder)
        num_of_files = 0
    print(f'{folder}: {num_of_files}')
    total += num_of_files


print(f'Not found folders: {not_found_folders}')
print(f'Number of not found folders {len(not_found_folders)}')
print(f'Total: {total}')
