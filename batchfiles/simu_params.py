'''
定义一些参数类，用来模拟原pipeline指令调用传递参数
'''
import numpy as np
import numba
import logging

# 将numba的日志等级提高到warning，从而屏蔽njit的超长log
logging.getLogger('numba').setLevel(logging.WARNING)
_logger = logging.getLogger(__name__)

class ParaPinhole:
    def __init__(self, focal_length=1446.238224784178, baseline=0.06):
        '''
        focal_length:针孔焦距
        pinhole_baseline:针孔基线
        '''
        self.focal_length = focal_length
        self.baseline = baseline

class ParaFisheye:
    def __init__(self, up_baseline=0.09, down_baseline=0.105, input_dir=None, output_dir=None, count=1, random_seed=0,
                 calibration_noise_level=0.0, rectify_config=None, keep_intermediate_images=False, fisheye_to_3pinholes=None, panorama_to_fisheye=None, remapping_folder=None):
        '''
        up_baseline:鱼眼上视基线
        down_baseline:鱼眼下视基线
        '''        
        self.up_baseline = up_baseline
        self.down_baseline = down_baseline
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.count = count
        self.random_seed = random_seed
        

class ParaStep2:
    def __init__(self, input_dir=None, output_dir=None, remappimg_folder=None, up_baseline=0.09, down_baseline=0.105,
                 keep_intermediate_images=False, rectify_config=None, calibration_noise_level=0.0, random_seed=0):
        '''
        input_dir:输入路径
        output_dir:输出路径
        remappimg_folder:
        rectify_config:path of the rectification config file
        keep_intermediate_images:Whether to keep the intermediate fisheye images
        '''
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.remappimg_folder = remappimg_folder
        self.up_baseline = up_baseline
        self.down_baseline = down_baseline
        self.keep_intermediate_images = keep_intermediate_images
        self.rectify_config = rectify_config
        self.calibration_noise_level = calibration_noise_level
        self.random_seed = random_seed
        
    def print(self):
        _logger.info(f'input_dir:{self.input_dir}, output_dir:{self.output_dir}, '
                     f'remappimg_folder:{self.remappimg_folder}, up_baseline:{self.up_baseline}, down_baseline:{self.down_baseline}')
        
class ParaFisheyeISP:
    def __init__(self, input_dir=None, output_dir=None, remappimg_folder=None, up_baseline=0.09, down_baseline=0.105,
                 keep_intermediate_images=False, rectify_config=None, calibration_noise_level=0.0, random_seed=0):
        '''
        input_dir:输入路径
        output_dir:输出路径
        remappimg_folder:
        rectify_config:path of the rectification config file
        keep_intermediate_images:Whether to keep the intermediate fisheye images
        '''
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.remappimg_folder = remappimg_folder
        self.up_baseline = up_baseline
        self.down_baseline = down_baseline
        self.keep_intermediate_images = keep_intermediate_images
        self.rectify_config = rectify_config
        self.calibration_noise_level = calibration_noise_level
        self.random_seed = random_seed    
if __name__ == "__main__":
    print('this is simu_params.py')