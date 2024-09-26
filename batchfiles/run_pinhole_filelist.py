import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

import lz4.frame as lz
import pickle as pkl
from multiprocessing import Pool, cpu_count
import multiprocessing
from subprocess import run
from functools import partial
import glob
import argparse
import distutils.spawn
import logging

import shutil

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

total_dir = r'/mnt/113-data/samba-share/simulation/filelist'
evalu_dir = '/mnt/119-data/samba-share/simulation/filelist/synt_pinhole_eval.csv'
total_train_dir = '/mnt/119-data/samba-share/simulation/filelist/synt_pinhole_randomTex.csv'

#total_val_dir = r'/mnt/113-data/samba-share/simulation/filelist/synt_pinhole_val_lz4.csv'     # 全景的val可以不用，津樑那边自己分
RELA_DIR = '/mnt/119-data/samba-share/simulation/evalue/'

focal_length=1446.238224784178  # 分辨率(1952，2784)pinhole相机的焦距

def generate_filelist(output_dir, results, disp_name):
    #用时间戳给filelist命名,可以不用删掉旧filelist直接生成新的filelist
    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    filelist_dir = output_dir / 'pinhole_filelist'
    filelist_dir.mkdir(parents=True, exist_ok=True)
    abs_output_dir = str(output_dir.absolute())
    filelist_path = filelist_dir / f'{disp_name}_filelist_{time_str}.csv'
    with open(filelist_path, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow([abs_output_dir] * 4)
        for item in results:            
            if item:
                # # 检查路径是否存在
                # p_exist = True
                # for i in item:
                #     i = Path(abs_output_dir) / Path(i)
                #     if not os.path.exists(i):
                #         print(f'{i} does not exist')
                #         p_exist = False
                #         break
                # if p_exist:
                    writer.writerow(item)
                
    # 将本地filelist跟总的filelist合并
    # 重新results,得到绝对路径
    absPath = Path(abs_output_dir)
    absResults = [(str((absPath / Path(item[0])).relative_to(RELA_DIR)), 
                   str((absPath / Path(item[1])).relative_to(RELA_DIR)),
                   str((absPath / Path(item[2])).relative_to(RELA_DIR)),
                   str((absPath / Path(item[3])).relative_to(RELA_DIR))) for item in results]
    
    return absResults, disp_name
    #migrate_filelist(absResults, disp_name)
                
def migrate_filelist(results, disp_name, filelist_dir):
    # 将本地filelist跟总的filelist合并
    # 根据视差文件名创建新的filelist，一般临时生成用

    totallist_path = filelist_dir
        
    with open(totallist_path, 'a') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in results:
            if item:
                writer.writerow(item)                    

#将已存在的filelist合并到总的filelist中
def migrate_filelist_by_filelist(fl_path):
    
    with open(fl_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
        pref = data[0][0] + '/'
        for i in range(1, len(data)):
            for j in range(0, 4):
                data[i][j] = pref + data[i][j]
                # 取相对路径
                rela_dir = RELA_DIR
                if rela_dir in data[i][j]:
                    data[i][j] = data[i][j].replace(rela_dir, '')
                else:
                    print(f'ERROR::{data[i][j]} does not contain {rela_dir}')
                    return 

    with open(total_train_dir, 'a') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in data[1:]:
            if item:
                writer.writerow(item)

def add_new_row(filelist_path, replay_str):
    # 读取第一列并拷贝
    with open(filelist_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    
    #新增一列
    newdata = [(item[0], item[1], item[2], item[0].replace('front', replay_str)) for item in data if len(item) >= 3]
    #new_filePath = filelist_path.replace('front', replay_str)    
    with open(filelist_path, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        fout.truncate(0)
        #writer.writerow([abs_output_dir] * 3)
        for item in newdata:
            if item:
                writer.writerow(item)
                
def remove_out_folder(path):
    # 获取path下所有包含'out'的文件夹
    folders = [str(p) for p in Path(path).rglob("*_out*") if p.is_dir()]
    # 删除这些文件夹
    for f in folders:
        print(f'remove {f}')
        os.system(f'rm -r {f}')
        
def read_lz4(lz4_file):
    if os.path.exists(lz4_file):
        with open(lz4_file, 'rb') as f:
            cdata = pkl.load(f)
        arr = lz.decompress(cdata['arr'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #print(f'decompressed {lz4_file}')
        return arr   

def _save_lz4(image_data, path):
    arr = np.ascontiguousarray(image_data)
    data = {
        'arr': lz.compress(arr, compression_level=3),
        'shape': image_data.shape,
        'dtype': image_data.dtype
    }
    with open(path, 'wb') as f:
        pkl.dump(data, f)
     
def pinhole_depth2disp(depth_file, output_dir):
    #if depth_file.suffix == '.lz4':
    depth_data = read_lz4(depth_file)
    
    if depth_data.shape != (2784, 1952):
        print(f"ERROR::pinhole shape is not (1952, 2784) but {depth_data.shape}")
        return
    
    baseline = 0.06
    #focal_length = args.focal_length
    disparity = baseline * focal_length / depth_data
    disparity = disparity.astype(np.float16)
    
    name = os.path.basename(depth_file)
    output_path = Path(output_dir) / name

    _save_lz4(disparity, output_path)

    print(f'Processed depth-to-disparity using file: {name}')    

def _run_exe(exe, input_dir, args):
    generate_disparity_exe = distutils.spawn.find_executable(exe)
    assert generate_disparity_exe, 'Error: executable `{exe}` is not available!'
    cmds = [generate_disparity_exe, f'--input-dir={input_dir}']
    _logger.info(f'Executing command:\n{" ".join(cmds)}')

    run(cmds)
            
def get_paths(output_dir, foldername = 'left'):
    pinhole_path = output_dir / foldername
    # 判断是否存在left文件夹
    if not pinhole_path.exists():
        return
    
    _logger.info(f'processing {pinhole_path}')
    # 获取视差文件夹
    #folders = [str(p) for p in Path(pinhole_path).rglob("*Disparity*") if p.is_dir()]
    disp_dir = pinhole_path / 'Disparity'
    if not disp_dir.exists():
        # 说明没有生成视差数据，需要生成
        disp_dir.mkdir(parents=True, exist_ok=True) # 创建视差文件夹
        #获取深度文件
        dp_path = pinhole_path / 'DepthPlanar'
        depth_files = [str(p) for p in dp_path.rglob("*.lz4") if p.is_file()]
        
        with Pool(multiprocessing.cpu_count()-1) as p:
            p.map(partial(pinhole_depth2disp, output_dir=str(disp_dir)), depth_files)    
        print(f'finished `{output_dir}` depth2disp')

    # 检查灰度图是否存在
    seg_dir = pinhole_path / 'Segmentation'
    files = [str(p) for p in seg_dir.rglob("*.png") if p.is_file()]
    if len(files) > 0:
        # 说明没有生成灰度图，需要生成
        #gary_dir.mkdir(parents=True, exist_ok=True)
        _run_exe("run_seg_gray_from_color.py", seg_dir, None)
    
    print(f'checked segment files {seg_dir}')
    
    disp_name = disp_dir.name
    right_name = 'right'
    
    # 获取视差文件夹下的所有文件的路径
    disp_files = [str(p) for p in disp_dir.rglob("*.lz4") if p.is_file()]
    # 获取相应的RGB和分割图文件的路径
    total_files = []
    for s in disp_files:
        # 检查路径是否存在
        p_limg = s.replace(disp_name, 'Scene').replace('.lz4', '.webp')
        p_disp = s
        p_gray = s.replace(disp_name, r'Segmentation/Graymap')
        p_rimg = p_limg.replace('left', right_name)
        if os.path.exists(p_limg) and os.path.exists(p_disp) and os.path.exists(p_gray) and os.path.exists(p_rimg):
            total_files.append((str(Path(p_limg).relative_to(output_dir)),
                                str(Path(p_disp).relative_to(output_dir)), 
                                str(Path(p_gray).relative_to(output_dir)), 
                                str(Path(p_rimg).relative_to(output_dir))))
        else:
            print(f'WARNING::file not ALL exist:{p_limg}')
    # 生成filelist
    return output_dir, total_files, foldername
    # generate_filelist(pinhole_path, total_files, disp_name)
    # _logger.info(f'generated {output_dir}')
        
def get_latest_pinhole_csv(dir):
    if not Path(dir).exists():
        print(f'{dir} does not exist')
        return False
    csv_files = [str(p) for p in Path(dir).rglob("*.csv") if p.is_file()]
    # 只返回最新的csv文件
    if len(csv_files) > 1:
        print(f'Warning:: {dir} has {len(csv_files)} csv files')
        latest_csv = max(csv_files, key=os.path.getctime)
        return latest_csv
    elif len(csv_files) == 0:
        print(f'Warning:: {dir} has no csv files')
        return False
    else:
        return csv_files[0]
        

def process_folder(folder_path, keyword = 'left'):
    '''处理一个文件夹下的所有pinhole数据，可以自动过滤已处理的子文件夹
    keyword: 用于检索的关键词，通常是left，但evo2等有其他表示方式，比如front_left
    '''
    dir = folder_path #'/mnt/119-data/samba-share/simulation/train/hypertiny_test/Autel_hyper_tinytest'
    folders = [str(p) for p in Path(dir).rglob(keyword) if p.is_dir()]
    print(f'The num of folder is {len(folders)}')

    n=0
    for f in folders:
        p = f.replace(keyword, '')
        fl_path = Path(p) / 'pinhole_filelist'
        # 过滤已合并过的filelist，避免二次合并
        n=n+1
        if fl_path.exists():
            #fl = glob.glob(str(fl_path) + '/*')
            #print(f'already migrate {fl_path}')
            _logger.warning(f'{n} already migrate {fl_path}')
            #migrate_filelist_by_filelist(fl[0])
            continue
        '''
        #evo2方式
        d = '/mnt/119-data/samba-share/simulation/filelist/synt_evo2_ph_front.csv'
        process_one_folder(p,'front_left', d)
        d = '/mnt/119-data/samba-share/simulation/filelist/synt_evo2_ph_up.csv'
        process_one_folder(p,'up_left', d)
        d = '/mnt/119-data/samba-share/simulation/filelist/synt_evo2_ph_down.csv'
        process_one_folder(p,'down_left', d)
        '''
        d = total_train_dir
        process_one_folder(p,'left', d)    
    print(n)

def process_one_folder(dir, foldername, filelist_dir):
    '''
    对于结构名称不一样的针孔数据集，比如EVO2,用foldername指定文件夹名称的方式生成filelist
    '''
    pinhole_path, total_files, disp_name = get_paths(Path(dir),foldername)
    absResults, disp_name = generate_filelist(pinhole_path, total_files, disp_name)
    migrate_filelist(absResults, disp_name, filelist_dir)    

if __name__ == "__main__":
    print('start')
    dir = '/mnt/119-data/samba-share/simulation/evalue'
    folders = [str(p) for p in glob.glob(dir+'/*/*') if Path(p).is_dir()]
    for f in folders:
        if 'renamed' in f:
            continue
        print(f)
        process_one_folder(f,'left', filelist_dir=evalu_dir)
