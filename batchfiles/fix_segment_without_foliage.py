import pickle as plk
import os
import numpy as np
import lz4.frame as lz
from loguru import logger
#import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing
from subprocess import run
from functools import partial

import shutil
import cv2
from PIL import Image
import sys
import csv
import glob
import time

#定义一个进度条
def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()

def load_pkl_img(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            cdata = plk.load(f)
        cdata = cdata['left_image']
        arr = lz.decompress(cdata['data'])
        #arr = lz.decompress(cdata['segmentation'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {pkl_file}')
        return arr
    
def load_pkl_seg(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            cdata = plk.load(f)
        cdata = cdata['segmentation']
        arr = lz.decompress(cdata['data'])
        #arr = lz.decompress(cdata['segmentation'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {pkl_file}')
        return arr    

def load_pkl_disp(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            cdata = plk.load(f)
        cdata = cdata['left_disparity']
        arr = lz.decompress(cdata['data'])
        #arr = lz.decompress(cdata['segmentation'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {pkl_file}')
        return arr   

def fix_segment(seg, disp):
    mask = disp > 0.1
    new_seg = seg.copy()
    new_seg[mask] = 1
    return new_seg

# 重新打包分割图
def pkl_seg(img_data, path):
    #np压缩成lz4
    carr = np.ascontiguousarray(img_data)   #先转成内存连续的数组才能用lz4压缩
    seg_data = {
        'segmentation': {'data': lz.compress(carr, 9), 'shape': img_data.shape, 'dtype': img_data.dtype},
        'default_shape': img_data.dtype
    }
    #保存,用plk打包
    with open(path, 'wb') as f:
        plk.dump(seg_data, f)

def process_pkl_seg(pkl_file):
    disp_path = pkl_file.replace('Segment', 'Disparity')
    seg = load_pkl_seg(pkl_file)
    disp = load_pkl_disp(disp_path)
    pkl_seg(fix_segment(seg, disp), pkl_file)
    logger.info(f'finish {pkl_file}')
        
def _multi_process_seg(folder):
    seg_files = [str(p) for p in Path(folder).rglob("*.pkl") if p.is_file()]
    with Pool(multiprocessing.cpu_count()-1) as p:
        p.map(partial(process_pkl_seg), seg_files)    
    logger.info(f'finish folder {folder}')

# 检查视差图是否有nan
def check_nan(file):
    disp = load_pkl_disp(file)
        
    #disp = load_pkl_disp(file)
    # 检查数组disp是否有nan
    if np.isnan(disp).any():
        #logger.info(f'nan in {file}')
        # 获得对应天空分割图
        seg_path = file.replace('Disparity', 'Segment')
        seg = load_pkl_seg(seg_path)
        sky_mask = seg == 0
        # 判断视差图为nan的位置是否对应天空分割图
        if sky_mask[np.isnan(disp)].all():
            logger.warning(f'TRUE---nan in {file} is sky')
        else:
            logger.error(f"FALSE---nan in {file} isn't all sky")
    # else:
    #     logger.info(f'no nan in {file}')
    
def load_lz4(lz4_file):
    if os.path.exists(lz4_file):
        with open(lz4_file, 'rb') as f:
            cdata = plk.load(f)
        arr = lz.decompress(cdata['arr'])
        arr = np.frombuffer(arr, dtype = cdata['dtype'])
        arr = np.reshape(arr, cdata['shape'])
        #print(arr)
        #logger.info(f'decompressed {lz4_file}')
        return arr  

def save_lz4(img_data, path):
        #np压缩成lz4
        carr = np.ascontiguousarray(img_data)   #先转成内存连续的数组才能用lz4压缩
        ldata ={
            'arr': lz.compress(carr, compression_level=3),
            'shape': img_data.shape,
            'dtype': img_data.dtype
        } 
        
        #检查深度是否有nan,有的话就不压缩成lz4,并保存到文件
        if np.isnan(img_data).any():
            logger.error(f'depth nan in {path}')
            return False
        #保存,用plk打包
        with open(path, 'wb') as f:
            plk.dump(ldata, f) 

        return True
    
def _multi_check_nan(folder):
    #检查pkl
    disp_files = [str(p) for p in Path(folder).rglob("*.pkl") if p.is_file()]
    with Pool(multiprocessing.cpu_count()-1) as p:
        p.map(partial(check_nan), disp_files)    
    logger.info(f'finish folder {folder}')   

def check_nan_lz4(file):
    disp = load_lz4(file)
    if np.isnan(disp).any():
        logger.error(f'nan in {file}')

def _multi_check_lz4_nan(folder):
    disp_files = [str(p) for p in Path(folder).rglob("*.lz4") if p.is_file()]
    with Pool(multiprocessing.cpu_count()-1) as p:
        p.map(partial(check_nan_lz4), disp_files)    
    logger.info(f'finish folder {folder}')   

#将文件夹转移到新文件夹路径下
def replace_folders(dir):
    #获取路径下所有带pickle和renamed的文件夹
    pkl_folders = [str(p) for p in Path(dir).rglob("*pickle*") if p.is_dir()]
    
    #获得dir的name
    dir_name = Path(dir).name
    #将文件夹转移到新文件夹路径下，新文件夹路径中的name替换为name+pickle
    for f in pkl_folders:
        new_f = f.replace(dir_name, dir_name+"_pickle")
        new_f = Path(new_f).parent
        # 假如新路径不存在则创建该新路径再转移
        if not Path(new_f).exists():
            Path(new_f).mkdir(parents=True, exist_ok=True)
        # 将f文件夹移动到new_f路径下
        #Path(f).replace(new_f)
        shutil.move(f, new_f)
        logger.info(f'finished move {f}')
        
    renamed_folders = [str(p) for p in Path(dir).rglob("*renamed*") if p.is_dir()]
    for f in renamed_folders:
        new_f = f.replace(dir_name, dir_name+"_out_renamed")
        new_f = Path(new_f).parent
        # 假如新路径不存在则创建该新路径再转移
        if not Path(new_f).exists():
            Path(new_f).mkdir(parents=True, exist_ok=True)
        # 将f文件夹移动到new_f路径下
        #Path(f).replace(new_f)
        shutil.move(f, new_f)
        logger.info(f'finished move {f}')

# 该方法旨在修复分割图没有后处理的情况
def fix_segment_label(path):
    # 先获取文件夹
    seg_folders = [str(p) for p in Path(path).rglob("*Segment*") if p.is_dir()]
    # 再获取文件
    for f in seg_folders:
        seg_pkls = [str(p) for p in Path(f).rglob("*.pkl") if p.is_file()]
        print(f'{len(seg_pkls)} in {f}')
        #continue
        with Pool(multiprocessing.cpu_count()-1) as p:
            p.map(partial(_remap_segmentation_id), seg_pkls)      

# 该方法旨在修复图片没有正确旋转90度的情况
def fix_shape_rot90_by_filelist(filelist_path, shape):
    # 获取全部文件
    lz4_files = []
    webp_files = []
    png_files = []

    print('loading files')
    abs_path = ''
    with open(filelist_path, 'r') as f:
        #获取每一行，逗号分隔
        reader = csv.reader(f, delimiter=',')
        rows = list(reader)
    
    if not Path(rows[0][0]).is_file():
        abs_path = rows[0][0]
    for row in rows:
        process_bar(rows.index(row), len(rows))
            
        for f in row:
            if abs_path != '':
                f = abs_path + '/' + f 
                
            if f.endswith(".lz4"):
                lz4_files.append(f)
            elif f.endswith(".webp"):
                webp_files.append(f)
            elif f.endswith(".png"):
                png_files.append(f)   

    print('checking lz4')
    wrong_lz = 0
    for f in lz4_files:
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        with open(f, 'rb') as f:
            cdata = plk.load(f)
            if cdata['shape'] != shape:
                wrong_lz += 1
                logger.info(f'wrong shape {f}')
                #重新旋转90
                img = load_lz4(f.name)
                img = np.rot90(img)
                save_lz4(img, f.name)
    
    logger.info(f'wrong lz4 {wrong_lz}')

    print('checking webp')
    i=0
    for f in webp_files:
        i+=1
        process_bar(i+1, len(webp_files))
        
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        # 读取webp
        img = Image.open(f)
        if img.size != (2560,5120):
            logger.info(f'wrong shape {f}')
            img=img.transpose(Image.ROTATE_90)
            img.save(f, lossless = True)
    
    print('checking png')
    i=0
    for f in png_files:
        i+=1
        process_bar(i+1, len(webp_files))
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        # 读取png
        img = Image.open(f)
        if img.size != (2560,5120):
            logger.info(f'wrong shape {f}')
            img=img.transpose(Image.ROTATE_90)
            img.save(f, lossless = True)

def fix_shape_rot90_by_cubefront(cubefront_path, shape):
    # 获取全部文件
    lz4_files = []
    webp_files = []
    png_files = []

    if not os.path.exists(cubefront_path):
        print(f'ERROR: {cubefront_path} not exist')
        
    files = [str(p) for p in Path(cubefront_path).rglob("*") if p.is_file()]
    i=0
    for f in files:
        i+=1
        process_bar(i, len(files))
                            
        if f.endswith(".lz4"):
            lz4_files.append(f)
        elif f.endswith(".webp"):
            webp_files.append(f)
        elif f.endswith(".png"):
            png_files.append(f)   

    print('checking lz4')
    wrong_lz = 0
    for f in lz4_files:
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        with open(f, 'rb') as f:
            cdata = plk.load(f)
            if cdata['shape'] != shape:
                wrong_lz += 1
                logger.info(f'wrong shape {f}')
                #重新旋转90
                img = load_lz4(f.name)
                img = np.rot90(img)
                save_lz4(img, f.name)
    
    logger.info(f'wrong lz4 {wrong_lz}')

    print('checking webp')
    i=0
    for f in webp_files:
        i+=1
        process_bar(i+1, len(webp_files))
        
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        # 读取webp
        img = Image.open(f)
        if img.size != (2560,5120):
            logger.info(f'wrong shape {f}')
            img=img.transpose(Image.ROTATE_90)
            img.save(f, lossless = True)
    
    print('checking png')
    i=0
    for f in png_files:
        i+=1
        process_bar(i+1, len(webp_files))
        if not os.path.exists(f):
            print(f'{f} not exist')
            continue
        # 读取png
        img = Image.open(f)
        if img.size != (2560,5120):
            logger.info(f'wrong shape {f}')
            img=img.transpose(Image.ROTATE_90)
            img.save(f, lossless = True)
                   
def _remap_segmentation_id(seg_path):
    seg_file = load_pkl_seg(seg_path)
    seg_graymap = seg_file.copy()
    seg_graymap[seg_graymap == 1] = 0
    seg_graymap[seg_graymap == 2] = 0
    seg_graymap[seg_graymap == 0] = 1
    seg_graymap[seg_graymap == 11] = 0
    seg_graymap[seg_graymap == 18] = 2
    pkl_seg(seg_graymap, seg_path)

def fixed_filelist_path_exist(filelist_path):
    with open(filelist_path, 'r') as f:
        #获取每一行，逗号分隔
        lines = f.readlines()
    
    rebuild = False
    badlilst = []
    for l in lines:
        process_bar(lines.index(l)+1, len(lines))
        line = l.strip() # 去掉每行头尾空白
        paths = line.split(',')
        for p in paths:
            if not os.path.exists(p):
                logger.error(f'{p} not exist')
                #记录错误的行
                badlilst.append(l)
                #lines.remove(l)
                rebuild=True
                break
    
    # 重写filelist
    if rebuild:
        for badline in badlilst:
            lines.remove(badline) 
        with open(filelist_path, 'w') as f:
            for line in lines:
                f.write(line)
        logger.info(f'rebuild {filelist_path}')

def fix_repeat_pkl_and_filelist():
    '''去重，因为某些意外导致同一批文件以新的文件夹生成了而不是覆盖原来的文件'''
    check_dir = '/mnt/119-data/R22612/Data/ERP/train/synthesis/'
    folders = glob.glob(check_dir + '*')
  
    # 对文件名进行处理
    for f in folders:
        f2 = f.replace('_synthesis', '_out_renamed_synthesis')
        if f2 in folders:
            
            #删除filelist中的这一行
            f2_name = Path(f2).name
            fl_path = '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/substation/pkl_filelist_rect_fine_train_substation_lz4y.csv'
            fl_path2 = '/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/substation/pkl_filelist_rect_fine_val_substation_lz4y.csv'
            fl_path3 = '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/fy_synt_0918_train_lz4y.csv'
            fl_path4 = '/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/fy_synt_0918_val_lz4y.csv'
            remove_specified_data_from_base_fl(fl_path, f2_name)
            remove_specified_data_from_base_fl(fl_path2, f2_name)
            remove_specified_data_from_base_fl(fl_path3, f2_name)
            remove_specified_data_from_base_fl(fl_path4, f2_name)
            
            fl_path5='/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/powerhouse/pkl_filelist_rect_fine_val_powerhouse_lz4y.csv'
            remove_specified_data_from_base_fl(fl_path5, f2_name)
            remove_specified_data_from_base_fl(fl_path5.replace('_val_','_train_'), f2_name)
            fl_path5='/mnt/119-data/R22612/Data/ERP/filelist/syn_filelist/tiny_val_lz4y.csv'
            remove_specified_data_from_base_fl(fl_path5, f2_name)
            remove_specified_data_from_base_fl(fl_path5.replace('_val_','_train_'), f2_name)
            fl_path5='/mnt/119-data/R22612/Data/ERP/filelist/lite_filelist/near/pkl_filelist_rect_fine_val_near_lz4y.csv'
            remove_specified_data_from_base_fl(fl_path5, f2_name)
            remove_specified_data_from_base_fl(fl_path5.replace('_val_','_train_'), f2_name)
            shutil.rmtree(f2)
            print(f'repeated {f2_name}')

# 该方法旨在将细小物体的数据从filelist中单独提取出来
def move_tiny_data_from_base_fl_to_tiny_fl(base_train_fl, tiny_train_fl, keyword):
    
    no_tiny_lines = []
    tiny_lines = []
    count = 0
    with open(base_train_fl,'r') as f:
        base_reader = csv.reader(f)
        for row in base_reader:
            # 将包含tiny的行写入目标
            if keyword in row[0]:                
                count+=1
                tiny_lines.append(row)
            # 保存不含tiny的行
            else:
                no_tiny_lines.append(row)
    
    # 重写不含tiny的行
    with open(base_train_fl, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(no_tiny_lines)
        
    # 将tiny的行写入tiny的filelist
    with open(tiny_train_fl, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(tiny_lines)
    
    print(f'{count} tiny frames')

# 根据关键词将指定的数据从filelist中移除
def remove_specified_data_from_base_fl(base_train_fl, keyword):
    bRepeat=False
    new_lines = []
    with open(base_train_fl,'r') as f:
        base_reader = csv.reader(f)
        for row in base_reader:
            # 将不包含keyword的行写入目标
            if keyword not in row[0]:                
               new_lines.append(row)
            else:
                bRepeat = True
    
    if not bRepeat:
        print(f'{keyword} not in {base_train_fl}')
        return 
    # 重写不含keyword的行
    with open(base_train_fl, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_lines)



def fix_segment_with_depth(file_folder):
    if not os.path.exists(file_folder):
        logger.error(f'{file_folder} not exist')
        return
    if ('left' or 'cube_front') not in file_folder:
        logger.error(f'{file_folder} not a correct folder')
        return
    
    seg_path = Path(file_folder) / 'Segmentation'
    seg_gray_path = seg_path / 'Graymap'
    seg_color_path = seg_path / 'Colormap'
    if not os.path.exists(seg_gray_path):
        print(f'{seg_gray_path} not exist')
        return
    # 获取全部分割图文件
    seg_gray_files = [str(p) for p in seg_gray_path.rglob("*lz4") if p.is_file()]
    
    with Pool(multiprocessing.cpu_count()-1) as p:
        p.map(partial(fix_one_seg_with_depth), seg_gray_files)

def fix_one_seg_with_depth(file_path):
    if 'left' in file_path:
        depth_path = file_path.replace('Segmentation/Graymap', 'DepthPlanar')
    elif 'cube_front' in file_path:
        depth_path = file_path.replace('CubeSegmentation/Graymap', 'CubeDepth')
         
    dp_file = load_lz4(depth_path)
    seg_gray_file = load_lz4(file_path)
    color_path = file_path.replace('Graymap', 'Colormap').replace('.lz4', '.webp')
    seg_color_file = cv2.imread(color_path)
    
    mask_sky = dp_file > 600
    new_seg_gray = seg_gray_file.copy()
    # 处理灰度图
    new_seg_gray[mask_sky] = 0
    save_lz4(new_seg_gray, file_path)
    # 处理RGB图
    new_seg_color = seg_color_file.copy()
    new_seg_color[mask_sky] = [70, 130, 180]
    new_seg_color = Image.fromarray(new_seg_color)
    new_seg_color.save(color_path, lossless=False)

# 细小物体filelist数量不对，检查一下用来合并的filelist数量是否正确
def check_add_filelistcount():
    dir = '/mnt/118-data/R22612/Data/ERP/train/synthesis/'
    folders = glob.glob(dir + '*')
    
    count = 0
    for f in folders:
        if 'hyper' in f:
            #count += 1
            fl_path = f + '/file_list/'
            fl = [str(p) for p in Path(fl_path).rglob("*.csv") if p.is_file()][0]
            with open(fl, 'r') as f:
                reader = csv.reader(f)
                lines = list(reader)
                count += len(lines)
    
    print(f'{count} tiny folders')

# 检查filelist总表文件对不对，删掉有问题或者搜不到的pkl
def check_filelist(filelist_dir):
    hasError = False
    rightRows = []
    with open(filelist_dir, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        abs_path = rows[0][0] + '/'
        rightRows.append(rows[0])
        for row in rows[1:]:
            process_bar(rows.index(row), len(rows))
            
            try:
                #分别读取disp,rpg,seg
                img = load_pkl_img(abs_path + row[0])
                disp = load_pkl_disp(abs_path + row[1])
                seg = load_pkl_seg(abs_path + row[2])
                
                if img is None:
                    hasError = True
                    logger.error(f'ERROR IN READING:{row[0]}')
                    continue
                if disp is None:
                    hasError = True
                    logger.error(f'ERROR IN READING:{row[1]}')
                    continue
                if seg is None:
                    hasError = True
                    logger.error(f'ERROR IN READING:{row[2]}')
                    continue
            
                                
            except Exception as e:
                hasError = True
                logger.error(f'ERROR Exception:{row[0]}:{e}')
                continue
            
            # 将正确的行写入新的filelist
            rightRows.append(row)
    
    # 如果有出错的，就删掉这行重写
    if hasError:
        print(f'fix {filelist_dir}')
        
        with open(filelist_dir, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rightRows)
    
    print(f'finish {filelist_dir}')
 
def fix_erp_diparity(disp_path):
    disp = load_lz4(disp_path)
    if not np.isnan(disp).any():
        return
    
    tmp_disp = disp.copy()
    tmp_disp[np.isnan(disp)] = 0
    # 重新压缩视差图
    save_lz4(tmp_disp, disp_path)            
 
def get_last_modified_time(file_path):
    ''' 获取文件的最后修改时间'''
    last_modified_time = os.path.getmtime(file_path)
    
    # 将时间戳转换为可读格式
    formatted_time = time.ctime(last_modified_time)
    
    return formatted_time 

def step3(d):
    folders = glob.glob(d + '/*')
    for f in folders:
        src = f + '/cube_128mm'
        shutil.rmtree(src)
    
    compare_folders = glob.glob(d + '_compare/*')
    for f in compare_folders:
        src = f + '/cube_rear'
        shutil.rmtree(src)
        cube_128 = f + '/cube_128mm'
        os.rename(cube_128, src)

              
if __name__ == '__main__':
    # d= '/mnt/119-data/samba-share/simulation/train/2x_1017'
    # folders = [str(p) for p in Path(d).rglob("Graymap_erp") if p.is_dir()]
    # for f in folders:
    #     print(f)
    #     seg_files = [str(p) for p in Path(f).rglob("*.lz4") if p.is_file()]
    #     for segfile in seg_files:
    #         seg = load_lz4(segfile)
    #         seg_cp = seg.copy()
    #         seg_cp[seg_cp == 1.5] = 2
    #         seg_cp[seg_cp == 0.75] = 1
    #         seg_cp=seg_cp.astype(np.uint8)
    #         save_lz4(seg_cp, segfile)
        
    d ='/mnt/119-data/samba-share/simulation/train/MX128/randomItem_pinhole/0712/'
    folders = [str(p) for p in Path(d).rglob("left") if p.is_dir()]
    print(len(folders))
    for f in folders:
        fix_segment_with_depth(f)
    exit(0)

    # 检查旋转错误
    folders = [str(p) for p in Path(d).rglob("cube_front") if p.is_dir()]
    print(len(folders))
    
    n=0
    for f in folders:
        n+=1
        disp_dir = Path(f) / 'Disparity'
        if disp_dir.exists():   #检查尚未处理过的文件夹
            continue
        print(f'{n}:{len(folders)}  {f}')
        fix_shape_rot90_by_cubefront(f, (5120,2560))
        fix_shape_rot90_by_cubefront(f.replace("cube_front", "cube_rear"), (5120,2560))
        fix_shape_rot90_by_cubefront(f.replace("cube_front", "cube_below"), (5120,2560))
    

            #_multi_process_seg(seg_path)

    
    # d = '/mnt/119-data/samba-share/simulation/train/hypertiny_test/Autel_hyper_tinytest/Autel_hyper_tinytest_2023-12-18-17-47-35_erp_pickle/Disparity/Autel_hyper_tinytest_2023-12-18-17-47-35_out_renamed/group0/cam0_0/Image_erp/1702892845073259776.pkl'
    # load_pkl_disp(d)
    #fix_segment_label(r'/mnt/118-data/R22612/Data/ERP/train/synthesis/')
    #replace_folders(r'/mnt/113-data/samba-share/simulation/train/2x_tiny')

    #fixed_filelist_path_exist(r'/mnt/113-data/samba-share/simulation/filelist/synt_1103_train_0.105_lz4.csv')
    # print('finish')
    # dir = r'/mnt/113-data/samba-share/simulation/train/2x_substation'
    # folder = [str(p) for p in Path(dir).rglob("*left*") if p.is_dir()]
    # for f in folder:
    #     fix_segment_with_depth(f)
    #     print(f'finish {f}')
    #fix_segment_with_depth(dir)
    #fix_shape_rot90('/mnt/113-data/samba-share/simulation/train/2x_tiny/2x_tiny_1023/2x_tiny_Autel_jungle/2x_tiny_Autel_jungle_2023-10-23-16-59-47/panorama_filelist/CubeDisparity_0.09_filelist_2023_12_20_08_54_40.csv', (5120,2560))
    # 113-data/samba-share/simulation/train/2x_lowfly_pickle_3/Segment/2x_lowfly_Autel_MountainGrassland_2023-09-27-14-32-21_renamed
    # folders = [str(p) for p in Path(dir).rglob("*pinhole_filelist*") if p.is_dir()]
    
    # for f in folders:
    #     #shutil.rmtree(f)
    #     os.system(f'rm -r {f}')
    #     print(f'removed {f}')
        
        
        
    # seg_path = r"D:\fix_seg\Segment\1695793546294420992.pkl"
    # process_pkl_seg(seg_path)
    # seg = load_pkl_seg(seg_path)
    # load the data
    # seg = load_pkl_seg(r"D:\fix_seg\seg\1695793546294420992.pkl")
    # disp = load_pkl_disp(dir)
    
