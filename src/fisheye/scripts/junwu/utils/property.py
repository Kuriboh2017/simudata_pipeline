import os 
from .dataio import load_pkl_img, load_pkl_disp, load_pkl_seg

def isvalid(path, type='img'):

    try:
        if type == 'img':
            limg, rimg = load_pkl_img(path)

        if type == 'disp':
            disp, error = load_pkl_disp(path)

        if type == 'seg':
            seg = load_pkl_seg(path)
    except:
        return False

    return True

def exists():
    pass