import cv2

def resize_960x704(img_in, is_disp=False, is_seg=False):
    assert img_in.shape[:2] in ((1069, 802),), 'resize input shape error'
    assert not (is_disp and is_seg), 'resize input error'
    if img_in.shape[:2] == (1069, 802):
        img_in = img_in[:, 9:-9, ...]
        img_out = cv2.resize(img_in,(704, 960),interpolation=cv2.INTER_LINEAR)

        if is_disp:
            img_out = img_out * img_out.shape[1] / img_in.shape[1]
        elif is_seg:
            img_out = img_out > 0.5
    return img_out

def resize_800x576(img_in, is_disp=False, is_seg=False):
    assert img_in.shape[:2] in ((1392, 976),), 'resize input shape error'
    assert not (is_disp and is_seg), 'resize input error'
    if img_in.shape[:2] == (1392, 976):
        img_in = img_in[18:-18, :, ...]
        img_out = cv2.resize(
            img_in,
            (576, 800),
            interpolation=cv2.INTER_LINEAR,
        )
        if is_disp:
            img_out = img_out * img_out.shape[1] / img_in.shape[1]
        elif is_seg:
            img_out = img_out > 0.5
    return img_out

def pad_704x1280(img_in, is_disp=False, is_seg=False, pad_type='reflect'):
    assert img_in.shape[:2] in ((681, 1280),), 'resize input shape error'
    assert not (is_disp and is_seg), 'resize input error'
    dh = 704 - img_in.shape[0]
    dw = 1280 - img_in.shape[1]

    if pad_type == 'reflect':
        img_out = cv2.copyMakeBorder(img_in, 0,dh,0,dw,borderType=cv2.BORDER_REFLECT)
    elif pad_type == 'zero':
        img_out = cv2.copyMakeBorder(img_in, 0,dh,0,dw,borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    return img_out