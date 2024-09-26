import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path
import glob
import shutil

''' 这批路径已经不用了
total_dir = r'/mnt/113-data/samba-share/simulation/filelist'
total_train_dir = r'/mnt/113-data/samba-share/simulation/filelist/synt_1103_train_baseline_lz4.csv'
total_val_dir = r'/mnt/113-data/samba-share/simulation/filelist/synt_1103_val_baseline_lz4.csv'     # 全景的val可以不用，津樑那边自己分
total_near105_dir = r'/mnt/113-data/samba-share/simulation/filelist/synt_near105_baseline_lz4.csv'
total_near135_dir = r'/mnt/113-data/samba-share/simulation/filelist/synt_near135_baseline_lz4.csv'
mh_panorama_water = '/mnt/119-data/samba-share/simulation/filelist/synt_panorama_water.csv'
mh_paronama_test = '/mnt/119-data/samba-share/simulation/filelist/test.csv'
'''
# 注意这个路径中的'baseline'是个变量，需要根据真实的baseline进行重命名
test_train_dir = r'/mnt/119-data/samba-share/simulation/filelist/synt_test_train_baseline_lz4.csv'  # 临时测试用
total_train_dir = r'/mnt/119-data/samba-share/simulation/filelist/synt_train_fisheye_baseline.csv'  # 24.9.11起使用的filelist
root_dir = '/mnt/119-data/samba-share/simulation/'

def generate_filelist(output_dir, results, disp_name):
    current_datetime = datetime.now()
    time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    filelist_dir = output_dir / 'panorama_filelist'
    filelist_dir.mkdir(parents=True, exist_ok=True)
    abs_output_dir = str(output_dir.absolute())
    filelist_path = filelist_dir / f'{disp_name}_filelist_{time_str}.csv'
    # 生成本地filelist
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
    # 重新results,得到相对路径
    absPath = Path(abs_output_dir.replace(root_dir, ''))
    absResults = [(str(absPath / Path(item[0])), str(absPath / Path(item[1])),
                   str(absPath / Path(item[2])), str(absPath / Path(item[3]))) for item in results]
    
    return absResults, disp_name
                
def migrate_filelist(results, disp_name):
    # 将本地filelist跟总的filelist合并

    # 根据baesline不同分别合并到不同的filelist中
    if '0.09' in disp_name:
        totallist_path = total_train_dir.replace('baseline', '0.09')
    elif '0.105' in disp_name:
        totallist_path = total_train_dir.replace('baseline', '0.105')
    elif '0.128' in disp_name:
        totallist_path = total_train_dir.replace('baseline', '0.128')
    elif '0.135' in disp_name:
        totallist_path = total_train_dir.replace('baseline', '0.135')
    else:
        print(f'ERROR:undefined baseline in {disp_name}')
        return
  
    with open(totallist_path, 'a') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in results:
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


def process_one_folder(input_dir):
    '''
    input_dir:必须是文件名/cube_front
    '''
    root = Path(input_dir).parent
    panorama_path = root / 'cube_front'
    # 跳过没后处理完的文件夹
    check_done_file = root / 'check.log'
    if not os.path.exists(check_done_file):
        return
    # 判断是否存在cube_front文件夹
    if not panorama_path.exists():
        return
    # 删除已经旧数据的filelist
    fl_path = root / 'panorama_filelist'
    if fl_path.exists():
        shutil.rmtree(fl_path)
    # 跳过已生成的文件
    if os.path.exists(Path(root) / 'fisheye_filelist_done.log'):
        return
        
    # 获取视差文件夹,一般是2个
    folders = [str(p) for p in Path(panorama_path).rglob("*Disparity*") if p.is_dir()]
    
    for f in folders:
        disp_name = Path(f).name
        # 根据baseline确定右视图
        if '0.09' in disp_name:
            right_name = 'rear'
        elif '0.105' in disp_name:
            right_name = 'below'
        elif '0.135' in disp_name:
            right_name = 'below' 
        elif '0.128' in disp_name:
            right_name = 'rear'
        else:
            right_name = 'rear'
            
        # 获取视差文件夹下的所有文件的路径
        disp_files = [str(p) for p in Path(f).rglob("*.lz4") if p.is_file()]
        # 获取相应的RGB和分割图文件的路径
        total_files = []
        for s in disp_files:
            # 检查路径是否存在
            p_limg = s.replace(disp_name, 'CubeScene').replace('.lz4', '.webp')
            p_disp = s
            p_gray = s.replace(disp_name, r'CubeSegmentation/Graymap')
            p_rimg = p_limg.replace('front', right_name)
            if os.path.exists(p_limg) and os.path.exists(p_disp) and os.path.exists(p_gray) and os.path.exists(p_rimg):
                total_files.append((str(Path(p_limg).relative_to(root)),
                                str(Path(p_disp).relative_to(root)), 
                                str(Path(p_gray).relative_to(root)), 
                                str(Path(p_rimg).relative_to(root))))
            else:
                print(f'WARNING::file not ALL exist:{p_limg}')
        
        # 生成本地filelist
        absResults, disp_name = generate_filelist(root, total_files, disp_name)
        print(f'generated {root}||{disp_name}')
        # 合并filelist到总filelist
        migrate_filelist(absResults, disp_name)
        
    # 生成filelist_done文件
    with open(Path(root) / 'fisheye_filelist_done.log', 'a') as f:
        f.write(f'{root} fisheye filelist done\n')    

def process_root_folder(root_dir):
    '''
    递归处理整个根目录下子文件夹的filelist
    '''
    folders = [str(p) for p in Path(root_dir).rglob("cube_front") if p.is_dir()]
    print(f'num of {root_dir} is {len(folders)}')
    for f in folders:
        process_one_folder(f) 

    
        
if __name__ == "__main__":
    file_name = '/mnt/119-data/samba-share/simulation/filelist/synt_train_fisheye_0.128.csv'
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # Replace the string in the data
    data = [[item.replace('/mnt/119-data/samba-share/simulation/', '') for item in row] for row in data]

    # Write the data back to the CSV file
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

        # for item in data:
        #     if item:
        #         writer.writerow(item)

