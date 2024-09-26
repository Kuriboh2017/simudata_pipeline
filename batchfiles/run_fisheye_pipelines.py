#!/usr/bin/env python3
import argparse
import distutils.spawn
import logging
import shutil
import glob
import os

from pathlib import Path
from subprocess import run

import run_seg_gray_from_color
import run_panorama_disparity_from_depth
import run_pinholes_disparity_from_depth
import simu_params  # 用模拟参数代替cmd传参，实现调用可控

import time

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)

kRemappingTablesDir = '/mnt/119-data/samba-share/simulation/code/2x'
#kRemappingTablesDir = r"G:\simu3\lens_effects\test_2x"
kDownsampledRootDir = '/mnt/119-data/R23024/data/junwu/data/synthesis'
kDownsampledRootDir_erp = '/mnt/119-data/R22612/Data/ERP/train/synthesis'


def _run_exe(exe, input_dir, args):
    generate_disparity_exe = distutils.spawn.find_executable(exe)
    assert generate_disparity_exe, 'Error: executable `{exe}` is not available!'
    cmds = [generate_disparity_exe, f'--input-dir={input_dir}']
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _run_pre_fisheye_steps(input_dir):
    '''
    Step 1: Preprocess: generate panorama disparity and segmentation graymap
      by running `run_panorama_disparity_from_depth.py` and 
      `run_seg_gray_from_color.py`
    '''
    input_dir = Path(input_dir)
    depth_dir = input_dir / 'cube_front' / 'CubeDepth'
    segment_dir = input_dir / 'cube_front' / 'CubeSegmentation'

    if segment_dir.exists():
        run_seg_gray_from_color.main(str(segment_dir))
    
    # ====== Generate Panorama Disparity ======
    if depth_dir.exists():
        if 'MX128' in str(input_dir):
            run_panorama_disparity_from_depth.main(str(depth_dir),up_baseline=0.128)
        elif 'MH136p6' in str(input_dir):
            run_panorama_disparity_from_depth.main(str(depth_dir),up_baseline=0.1366, down_baseline=0.1366)
        else:
            run_panorama_disparity_from_depth.main(str(depth_dir))
    # ====== Generate Panorama Disparity ======

    # pinhole现在也需要视差了，如果不用可以在这里注释掉
    '''
    '''
    ph_dp_dir = [str(p) for p in input_dir.rglob("DepthPlanar") if p.is_dir()] #input_dir / 'left' / 'DepthPlanar'
    #ph_seg_dir = input_dir / 'left' / 'Segmentation'
    for ph_dir in ph_dp_dir:
        if not Path(ph_dir).exists():
            return
        if 'front_left' in ph_dir:
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=0.06, focal_length=834.0642386183716)
        elif 'up_left' in ph_dir:
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=0.04, focal_length=662.7394008259646)
        elif 'down_left' in ph_dir:
            run_pinholes_disparity_from_depth.main(ph_dir, baseline=0.08, focal_length=662.7394008259646)
        else:
            run_pinholes_disparity_from_depth.main(ph_dir)
            
        run_seg_gray_from_color.main(ph_dir.replace("DepthPlanar",'Segmentation'))

def _run_pre_fisheye_steps_dilation(input_dir):
    '''这是一个修复函数，旨在对已生成的全景和针孔图的视差和分割图进行膨胀'''
    _logger.info(f'Processing {input_dir}')
    input_dir = Path(input_dir)
    disp_dir1 = input_dir / 'cube_front' / 'CubeDisparity_0.09'
    disp_dir2 = input_dir / 'cube_front' / 'CubeDisparity_0.105'
    segment_dir = input_dir / 'cube_front' / 'CubeSegmentation' / 'Graymap'

    
    run_seg_gray_from_color.tmp_dilate_seg(str(segment_dir))
    
    # ====== Generate Panorama Disparity ======
    run_panorama_disparity_from_depth.tmp_dilation_disp(str(disp_dir1))
    run_panorama_disparity_from_depth.tmp_dilation_disp(str(disp_dir2))
    # ====== Generate Panorama Disparity ======

    # pinhole现在也需要视差了，如果不用可以在这里注释掉
    '''
    因为权限问题，针孔要用sim跑，视差要用23091跑
    '''
    ph_dp_dir = input_dir / 'left' / 'Disparity'
    ph_seg_dir = input_dir / 'left' / 'Segmentation' / 'Graymap'
    if not ph_dp_dir.exists():
        return
    run_panorama_disparity_from_depth.tmp_dilation_disp(str(ph_dp_dir))
    run_seg_gray_from_color.tmp_dilate_seg(str(ph_seg_dir)) 
    
    _logger.info(f'Finished dilation: {input_dir}')   

def _run_fisheye_steps(args):
    '''
    Step 2: Fisheye, ISP, Camera noises by running `run_fisheye_isp_combo_2x_by_folder.py`
    '''
    import run_fisheye_isp_combo_2x_by_folder
    # fisheye_exe = distutils.spawn.find_executable(
    #     'run_fisheye_isp_combo_2x_by_folder.py')
    # assert fisheye_exe, 'Error: python executable `run_fisheye_isp_combo_2x_by_folder.py` is not available!'
    if args.output_dir is None:
        input_dir = Path(args.input_dir)    
        output_dir = input_dir.parent / f'{input_dir.name}_out'
    else:
        output_dir = Path(args.output_dir)
        
    step2args = simu_params.ParaStep2() # 用新的参数
    step2args.input_dir = args.input_dir
    step2args.output_dir = str(output_dir)
    step2args.remappimg_folder = kRemappingTablesDir
    step2args.up_baseline = args.up_baseline
    step2args.down_baseline = args.down_baseline
    
    step2args.print()
    run_fisheye_isp_combo_2x_by_folder._main(step2args)

    return output_dir


def _delete_cubedisparity_subdirs(parent_dir, prefix="CubeDisparity_"):
    parent_dir_path = Path(parent_dir)
    if not parent_dir_path.exists():
        _logger.error(f"Error: The directory {parent_dir} does not exist.")
        return
    if not parent_dir_path.is_dir():
        _logger.error(f"Error: {parent_dir} is not a directory.")
        return
    for subdir in parent_dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith(prefix):
            try:                
                shutil.rmtree(subdir)
                _logger.info(f"Deleted {subdir}")
            except Exception as e:
                _logger.info(f"Error deleting {subdir}: {e}")


def _remove_disparity(args):
    '''
    Step 2.1: Remove the disparity
    '''
    target_dir = Path(args.input_dir) / 'cube_front'
    _delete_cubedisparity_subdirs(target_dir, args)


def _run_post_fisheye_steps(fisheye_output_dir, args):
    '''
    Step 3: Postprocess: generate disparity and segmentation graymap
      Rename to group0/group1, Rotate up-fisheye images 180 degrees, etc.
    '''
    exe = "cube3_post_fisheye_process.py"
    post_fisheye_renaming = distutils.spawn.find_executable(exe)
    assert post_fisheye_renaming, 'Error: executable `{exe}` is not available!'
    renamed_dir = fisheye_output_dir.parent / \
        f'{fisheye_output_dir.name}_renamed'
    cmds = [
        post_fisheye_renaming,
        f'--input-dir={fisheye_output_dir}',
        f'--output-dir={renamed_dir}',
        f'--up-baseline={args.up_baseline}',
        f'--down-baseline={args.down_baseline}',
    ]
    if args.rectify_config is not None and args.rectify_config != 'None':
        cmds.append(f'--rectify-config={args.rectify_config}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return renamed_dir


def _generate_pickle_files(renamed_dir, args):
    '''
    Step 4: Generate the pickle files by running `run_sim_to_perception_pickle.py`
    TODO: convert ERP to pickle as well.
    '''
    exe = "run_sim_to_perception_pickle.py"
    generate_pickle_exe = distutils.spawn.find_executable(exe)
    assert generate_pickle_exe, 'Error: executable `{exe}` is not available!'
    cmds = [generate_pickle_exe,
            f'--input-dir={renamed_dir}',
            f'--up-baseline={args.up_baseline}',
            f'--down-baseline={args.down_baseline}',
            ]
    input_dir = Path(args.input_dir)
    pickle_dir = input_dir.parent / f'{input_dir.name}_pickle'
    cmds.append(f'--output-dir={pickle_dir}')
    downsampled_dir = Path(kDownsampledRootDir) / \
        f'add_{input_dir.name}_synthesis'
    cmds.append(f'--downsampled-output-dir={downsampled_dir}')
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)
    return downsampled_dir


def _combine_pickle_files(downsampled_dir, args):
    '''
    Step 5: Combine the pickle files by running `migrate_filelist.py`
    '''
    exe = "migrate_filelist.py"
    combine_pickle_exe = distutils.spawn.find_executable(exe)
    assert combine_pickle_exe, 'Error: executable `{exe}` is not available!'
    filelist_dir = Path(downsampled_dir) / 'file_list'
    files = list(filelist_dir.glob('pkl_filelist_fisheye*.csv'))
    if args.dry_run:
        src_filelist = 'dry_run'
    else:
        assert (
            files
        ), f'Error: {len(files)} pkl_filelist_fisheye*.csv files found in {filelist_dir}'
        src_filelist = Path(files[0]).absolute()
    # 根据文件夹名字临时判断是给谁更新
    if 'substation' in str(downsampled_dir):
        '''变电站路径'''
        cmds = [
            combine_pickle_exe,
            f'--src-filelist={src_filelist}',
            '--dst-train-filelist=/mnt/112-data/R23024/data1/junwu/tea_gi/data_transform/update_auto_detect/filelist/lite_filelist/substation/pkl_filelist_rect_fine_train_substation_lz4y.csv',
            '--dst-eval-filelist=/mnt/112-data/R23024/data1/junwu/tea_gi/data_transform/update_auto_detect/filelist/lite_filelist/substation/pkl_filelist_rect_fine_val_substation_lz4y.csv',
        ]
    else:
        cmds = [
            combine_pickle_exe,
            f'--src-filelist={src_filelist}',
            '--dst-train-filelist=/mnt/112-data/R23024/data/junwu/data/filelist/synt_0918_train_lz4y.csv',
            '--dst-eval-filelist=/mnt/112-data/R23024/data/junwu/data/filelist/synt_0918_val_lz4y.csv',
        ]
    '''评测路径'''
    # cmds = [
    #     combine_pickle_exe,
    #     f'--src-filelist={src_filelist}',
    #     '--dst-train-filelist=/mnt/112-data/R23024/data/junwu/data/filelist/synt_0918_test_lz4y.csv',
    #     '--dst-eval-filelist=/mnt/112-data/R23024/data/junwu/data/filelist/synt_0918_test_lz4y.csv',
    # ]
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _main(args):
    # Step 1: pre_fisheye, disparity, segmentation
    _run_pre_fisheye_steps(args.input_dir)
    _logger.info('Step1: pre_fisheye_steps done!')

    # Step 2: fisheye, isp, noises,
    fisheye_output_dir = _run_fisheye_steps(args)
    _logger.info('Step2: fisheye_steps done!')
    _logger.info(f'fisheye_output_dir={fisheye_output_dir}')
    # input_dir = Path(args.input_dir)
    # fisheye_output_dir = input_dir.parent / f'{input_dir.name}_out'

    # # Step 3: post_fisheye
    # renamed_dir = _run_post_fisheye_steps(fisheye_output_dir, args)
    # _logger.info('Step3: post_fisheye_steps done!')
    # _logger.info(f'renamed_dir={renamed_dir}')

    # # Step 4: generate pickle files
    # downsampled_dir = _generate_pickle_files(renamed_dir, args)
    # _logger.info('Step4: Generated pickle files done!')

    # # Step 5: Combine the pickle files
    # _combine_pickle_files(downsampled_dir, args)
    # _logger.info('Step5: Combined pickle files! All done')

    # jinliang的erp训练需要全景图的视差图，所以就不删了
    # _remove_disparity(args)
    # _logger.info('Removed intermediated CubeDisparity!')

    # try:
    #     if not args.dry_run:
    #         shutil.rmtree(fisheye_output_dir)
    #     _logger.info(f"Deleted {fisheye_output_dir}")
    # except Exception as e:
    #     _logger.info(f"Error deleting {fisheye_output_dir}: {e}")

def batch_run_paranoma_pinhole(folder):
    # 一级目录
    folders = glob.glob(folder + '/*')
    n = 0
    for folder in folders:
        if Path(folder).is_dir():
            subfolders = glob.glob(folder + '/*') 
            for subf in subfolders:
                if Path(subf).is_dir():
                    if 'renamed' in subf or 'pickle' in subf:
                        continue
                    n+=1
                    _logger.info(f'{n} process {subf}')
                    #_run_pre_fisheye_steps(folder)
                    _run_pre_fisheye_steps_dilation(subf)
        
def get_last_modified_time(file_path):
    ''' 获取文件的最后修改时间'''
    last_modified_time = os.path.getmtime(file_path)
    
    # 将时间戳转换为可读格式
    formatted_time = time.ctime(last_modified_time)
    
    return formatted_time 

# 从整个大文件夹筛选出需要处理的文件夹，并处理step1
def process(folder, filter_str='cube_front'):
    d= folder
    # 获取待处理的文件夹
    folders = [str(p) for p in Path(d).rglob(filter_str) if p.is_dir()]
    print(len(folders))
    
    n=0
    for f in folders:
        f = f.replace(filter_str, '')
        if os.path.exists(f):
            n += 1
            _logger.info(f'{n}:{len(folders)} processing {f}')
            _run_pre_fisheye_steps(f)
            
def rm_disparity(folder):
    '''删除视差图'''
    folders = [str(p) for p in Path(folder).rglob('cube_front') if p.is_dir()]
    for f in folders:
        
        _delete_cubedisparity_subdirs(f)
            
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run fisheye ISP pipeline')
    # parser.add_argument('-i', '--input-dir', required=True,
    #                     help='Input directory')
    # parser.add_argument('-o', '--output-dir', required=False,
    #                     help='Output directory')
    # parser.add_argument('-r', '--rectify-config',
    #                     help='path of the rectification config file')
    # parser.add_argument('-ub', '--up-baseline', type=float, default=0.09,
    #                     help='up camera baseline')
    # parser.add_argument('-db', '--down-baseline', type=float, default=0.105,
    #                     help='down camera baseline')
    # parser.add_argument('--random-roll-max', type=float, default=0.0,
    #                     help='use a random roll angle in degrees')
    # parser.add_argument('-d', '--dry-run', action='store_true',
    #                     help='Dry run')
    # args = parser.parse_args()
    # _main(args)
    #batch_run_paranoma_pinhole('/mnt/119-data/samba-share/simulation/train/hypertiny_test')
    
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1007"
    # rm_disparity(d)
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1017"
    # rm_disparity(d)
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1024"
    # rm_disparity(d)    
    # d= "/mnt/119-data/samba-share/simulation/train/2x_1116"
    # rm_disparity(d) 
    
    d=r"D:\RecordedData\2x_MX128_Autel4_randomEmissive_2024-08-06-15-31-51"
    # disp = [str(p) for p in Path(d).rglob("Disparity") if p.is_dir()]
    # for f in disp:
    #     shutil.rmtree(f)
    #     print(f'Deleted {f}')
    print(d)
    process(d,'left')
