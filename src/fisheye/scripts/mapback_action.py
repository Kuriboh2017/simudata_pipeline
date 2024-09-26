from multiprocessing import Pool, cpu_count
import shutil
import numpy as np

data = np.load('by_file_src_to_dst_without_seg.npz',
               allow_pickle=True)['src_to_dst']
data_item = data.item()

src_to_dst_arr = np.array(list(data_item.items()))


def _copy_file(src_to_dst):
    src, dst = src_to_dst
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(e)
        print('ERROR: ', src, dst)


with Pool(cpu_count()) as p:
    p.map(_copy_file, src_to_dst_arr)
