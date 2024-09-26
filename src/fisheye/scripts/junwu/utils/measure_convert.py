import hydra
from omegaconf import DictConfig

# @hydra.main(version_base=None, config_path='config', config_name='evaluate.yaml')
def disp_to_depth(cfgs, disp):
    # depth    = np.zeros_like(disp)
    disp_max = 256 # cfgs.camera_params.baseline * cfgs.camera_params.fx / 0.5 # depth on 0.5m
    disp_min = cfgs.camera_params.baseline * cfgs.camera_params.fx / 60 # depth on 70m
    disp[disp > disp_max] = disp_max
    disp[disp < disp_min] = disp_min

    depth = cfgs.camera_params.baseline * cfgs.camera_params.fx / disp
    return depth

