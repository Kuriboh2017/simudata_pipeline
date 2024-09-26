import re
import cv2
import struct

import numpy as np

from .sobel import sobel_mask_cv
from .pointcloud import disp2pointcloud_fisheye, disp2pointcloud_pinhole


def lr_check_err(img, ldisp, rdisp):
    def get_grid(img):
        H, W, C = img.shape
        x = np.arange(0, W)
        y = np.arange(0, H)
        X, Y = np.meshgrid(x, y)
        return X

    xcoor_grid = get_grid(img)
    xcoor_recon_grid = xcoor_grid - ldisp + rdisp
    lr_check_err = abs(xcoor_recon_grid - xcoor_grid)
    lderr = lr_check_err

    return lderr


def min_lrcerror_merge(disp_lst, lrerr_lst):
    # merge disp from large models of different versions
    disps_arr = np.stack(disp_lst)
    errs_arr  = np.stack(lrerr_lst)
    sel_idx = np.expand_dims(np.argmin(errs_arr, axis=0), axis=0)
    disp_min_err = np.squeeze(np.take_along_axis(disps_arr, sel_idx, axis=0))
    derr_min_err = np.min(errs_arr, axis=0)

    sel_idx_map = np.squeeze(sel_idx)
    # sel_idx_map = ((sel_idx_map+1) / 5)*255
    # sel_idx_map = cv2.applyColorMap(sel_idx_map.astype(np.uint8), cv2.COLORMAP_HSV)

    return disp_min_err, derr_min_err, sel_idx_map


def merge(disps_derrs, lage_seg):

    disp_lage, derr_lage = disps_derrs['large']    # Disp ID 0
    disp_lite, derr_lite = disps_derrs['lite']     # Disp ID 1
    disp_opcv, derr_opcv = disps_derrs['opencv']   # Disp ID 2
    disp_3d,   derr_3d   = disps_derrs['3drecon']  # Disp ID 3
    sel_idx_map = np.zeros_like(disp_lage)


    # mix large disp and lite disp
    lite_mask = derr_lite < derr_lage
    disp_mix  = (~lite_mask) * disp_lage + lite_mask * disp_lite
    derr_mix  = (~lite_mask) * derr_lage + lite_mask * derr_lite
    sel_idx_map[lite_mask] = 1

    # mix opencv disp with previous processed disp
    # if self.opencv_dispErr:
    #     mask_opcv = np.logical_and(disp_opcv>=0, derr_mix>self.opencv_dispErr_threshold)
    # else:
    mask_opcv = disp_opcv>0
    mask_opcv = np.logical_and(mask_opcv, lage_seg!=0) # remove sky, but sky will set to 0 later
    mask_opcv = np.logical_and(mask_opcv, derr_mix<1000) # remove disable pixels
    mask_opcv = np.logical_and(mask_opcv, disp_mix>0.88) # more than 40m, opencv is not statble for training.boundary

    disp_mix[mask_opcv] = disp_opcv[mask_opcv]
    # derr_mix[mask_opcv] = 0.002
    sel_idx_map[mask_opcv] = 2

    # merge with disp_3d 
    # derr_3d[disp_3d <= 0]  = 1000
    # derr_3d[disp_3d >= 0] = 0.001
    # disp_lst = [disp_mix, disp_3d]
    # derr_lst = [derr_mix, derr_3d]
    # disp_mix, derr_mix, sel_idx_map_tmp = min_lrcerror_merge(disp_lst, derr_lst)
    mask_3d = disp_3d > 0.001
    mask_3d = np.logical_and(mask_3d, derr_3d < 500) # remove disable pixels
    mask_3d = np.logical_and(mask_3d, derr_mix<1000) # remove disable pixels
    mask_3d = np.logical_and(mask_3d, disp_mix>0.88) # more than 40m, opencv is not statble for training.boundary
    disp_mix[mask_3d] = disp_3d[mask_3d]
    sel_idx_map[mask_3d] = 3


    # wire enhancement by using the labels from lite model
    wire_mask = lage_seg == 2
    disp_mix  = (~wire_mask) * disp_mix + wire_mask * disp_lite
    derr_mix  = (~wire_mask) * derr_mix + wire_mask * derr_lite
    sel_idx_map[wire_mask] = 1

    return disp_mix, derr_mix, sel_idx_map.astype(np.uint8)


def read_npz(path):
    try:
        data = np.load(path, allow_pickle=True)
        data = data[list(data.keys())[0]]
    except Exception as e:
        print('[Error] NPZ File:', path)
        print(e)
    return data

def read_tradition_disp(path):
    try:
        data = dict(np.load(path, allow_pickle=True))
    except:
        data = {}
    return data

def read_disp_npz(npz_path):
    disp_data = np.load(npz_path)

    disp = disp_data['disp'].astype(np.float32)
    seg = disp_data['seg_idx']
    uncertainty = disp_data['uncert'].astype(np.float32)
    return disp, seg, uncertainty

def read_disp(path):
    try:
        data = np.load(path, allow_pickle=True)
        if 'error' in list(data.keys()):
            disp = data['disp']
            error = data['error']
        else:
            disp = data[list(data.keys())[0]]
            error = disp * 0
    except Exception as e:
        print('[Error] NPZ File:', path)
        print(e)
    return disp.astype('float32'), error.astype('float32')

def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # data = np.flipud(data)
    file.close()
    return data * scale

def read_dat(
    path,
    headSize=5656,
    rows=800, # h 704 x 960 fisheye
    cols=576, # w
    step=3,
    bodyBsize=None,
    dispBsize=None,
    pcBsize=None,
    dispRows=None,
    dispCols=None,
):

    # rows = 640
    # cols = 480
    # step = 3
    # bodyBsize=rows*cols*step
    # headSize=5656
    # dispBsize= (rows//4)*(cols//4)*4

    if bodyBsize is None:
        bodyBsize = rows*cols*step
    if dispBsize is None:
        dispBsize = (rows//4)*(cols//4)*4
    if pcBsize is None:
        pcBsize = (rows//4)*(cols//4)*4*4
    if dispRows is None:
        dispRows = (rows//4)
    if dispCols is None:
        dispCols = (cols//4)

    with open(path, mode="rb") as file:
        contents = file.read()

    BinoParam = contents[1204:2940]
    intrinsic = BinoParam[204:236]
    trans = BinoParam[236:260]

    rect1 = contents[headSize:(headSize+bodyBsize)]
    rect2 = contents[(headSize+bodyBsize):(headSize+bodyBsize*2)]
    dispmap = contents[(headSize+bodyBsize*2):(headSize+bodyBsize*2+dispBsize)]
    point_cloud = contents[(headSize+bodyBsize*2+dispBsize):(headSize+bodyBsize*2+dispBsize+pcBsize)]

    intrinsic = np.frombuffer(intrinsic, dtype=np.float64)
    trans = np.frombuffer(trans, dtype=np.float64)

    left = np.frombuffer(rect1, dtype=np.uint8)
    left = left.reshape(rows, cols, step)

    right = np.frombuffer(rect2, dtype=np.uint8)
    right = right.reshape(rows, cols, step)

    disp = np.frombuffer(dispmap, dtype=np.float32)
    disp = disp.reshape(dispRows, dispCols)

    point_cloud = np.frombuffer(point_cloud, dtype=np.float32)
    point_cloud = point_cloud.reshape(dispRows, dispCols, 4)

    return left, right, disp, intrinsic, trans, point_cloud

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


def disp2edgemask(disp, kernel, camera_mode, fx, fy, baseline):
    disp_f = cv2.dilate(disp, kernel)

    if camera_mode == 'pinhole':
        point_cloud_f = disp2pointcloud_pinhole(
            disp_f, fx=fx/ 4.0, fy=fy / 4.0, cx=disp.shape[1] / 2,
            cy=disp.shape[0] / 2, baseline=baseline, max_depth=100000
        )
    else:
        point_cloud_f = disp2pointcloud_fisheye(
            disp_f, fx=fx / 4.0, fy=fy / 4.0, cx=disp.shape[1] / 2,
            cy=disp.shape[0] / 2, baseline=baseline, max_depth=100000
        )
    depth_f = point_cloud_f[:, :, -1]
    edge_mask = sobel_mask_cv(depth_f, 15)

    return edge_mask

class BinoParam:
    def __init__(self, data):
        # BinoParam = contents[1204:2940]  #
        # intrinsic = BinoParam[204:236]  # 32
        # trans = BinoParam[236:260]  # 24

        offset = 4  # 前面4个字节标识calibmean 标定方式
        (self.srcRows, self.srcCols, self.dstRows, self.dstCols) = struct.unpack('4i', data[offset:offset + 16])
        # print("srcRows: {}, srcCols: {}, dstRows: {}, dstCols: {}".format(self.srcRows, self.srcCols, self.dstRows,
        #                                                                   self.dstCols))
        offset = 24
        self.srcParam1 = struct.unpack('9d', data[offset:offset + 72])
        # print("srcParam1: {}".format(self.srcParam1))

        offset = 96
        self.srcParam2 = struct.unpack('9d', data[offset:offset + 72])
        # print("srcParam2: {}".format(self.srcParam2))

        offset = 168
        self.dstParam1 = struct.unpack('4d', data[offset:offset + 32])  # FX FY CX CY
        # print("dstParam1: {}".format(self.dstParam1))

        offset = 200
        self.dstParam2 = struct.unpack('4d', data[offset:offset + 32])
        # print("dstParam2: {}".format(self.dstParam2))


class Mat2D:
    def __init__(self, name, data):
        self.name = name
        self.rows, = struct.unpack('i', data[0:4])
        self.cols, = struct.unpack('i', data[4:8])
        self.chns, = struct.unpack('i', data[8:12])
        self.step, = struct.unpack('i', data[12:16])
        self.ptype, = struct.unpack('i', data[16:20])
        self.color, = struct.unpack('i', data[20:24])
        self.frame, = struct.unpack('i', data[24:28])
        self.datasz, = struct.unpack('i', data[28:32])
        #
        # print("{} rows: {} cols: {} chns: {} step: {} type: {} color: {} frame: {} datasz: {}".format(self.name,
        #                                                                                               self.rows,
        #                                                                                               self.cols,
        #                                                                                               self.chns,
        #                                                                                               self.step,
        #                                                                                               self.ptype,
        #                                                                                               self.color,
        #                                                                                               self.frame,
        #                                                                                               self.datasz))


class AIOHead:
    def __init__(self, data):
        # binoParam
        offset = 1208
        length = 1736
        self.binoParam = BinoParam(data[offset:offset + length])

        # mats
        offset = 4152
        self.dist = Mat2D("dist", data[offset:offset + 96])
        offset = offset + 96
        self.dist1 = Mat2D("dist1", data[offset:offset + 96])
        offset = offset + 96
        self.dist2 = Mat2D("dist2", data[offset:offset + 96])
        offset = offset + 96
        self.rect = Mat2D("rect", data[offset:offset + 96])
        offset = offset + 96
        self.rect1 = Mat2D("rect1", data[offset:offset + 96])
        offset = offset + 96
        self.rect2 = Mat2D("rect2", data[offset:offset + 96])
        offset = offset + 96
        self.disp = Mat2D("disp", data[offset:offset + 96])
        offset = offset + 96
        self.depth = Mat2D("depth", data[offset:offset + 96])
        offset = offset + 96
        self.cloud = Mat2D("cloud", data[offset:offset + 96])
        offset = offset + 96
        self.flow = Mat2D("flow", data[offset:offset + 96])
        offset = offset + 96
        self.segmn = Mat2D("segmn", data[offset:offset + 96])
        offset = offset + 96
        self.mat2d = Mat2D("mat2d", data[offset:offset + 96])


# 至少有四种类型数据需要测试确认：飞机上采集的不含原图的、飞机采集的含原图的、标定工具采集的来源于标定模式、标定工具采集的来源于感知测试工具
def Parser(fpath):
    fp = open(fpath, 'rb')
    try:
        data = fp.read()
    finally:
        fp.close()

    # parser header
    header = data[:5656]

    aioHead = AIOHead(header)
    # parseHeader(header, aioHead)
    # parse content
    headSize = 5656
    if aioHead.rect.datasz != 0:
        rectstart = headSize + aioHead.dist.datasz + aioHead.dist1.datasz + aioHead.dist2.datasz
        rect1 = data[rectstart:rectstart + aioHead.rect.datasz]
        LR = np.frombuffer(rect1, dtype=np.uint8)
        LR = LR.reshape(aioHead.rect.rows, aioHead.rect.cols, aioHead.rect.chns)

        dist_cols = int(aioHead.dist.cols / 2)
        image_value = data[headSize:headSize + aioHead.dist.datasz]
        image = np.frombuffer(image_value, dtype=np.uint8)
        image = image.reshape(aioHead.dist.rows, aioHead.dist.cols, aioHead.dist.chns)

        image_cols = int(aioHead.rect.cols / 2)
        left, right = LR[:, :image_cols, :], LR[:, image_cols:, :]
        dist1_image_left, dist2_image_right = image[:, :dist_cols, :], image[:, dist_cols:, :]

    if aioHead.rect.datasz == 0:
        rectstart_left = headSize + aioHead.dist.datasz + aioHead.dist1.datasz + aioHead.dist2.datasz + aioHead.rect.datasz
        rect1_left = data[rectstart_left:rectstart_left + aioHead.rect1.datasz]
        left = np.frombuffer(rect1_left, dtype=np.uint8)
        left = left.reshape(aioHead.rect1.rows, aioHead.rect1.cols, aioHead.rect1.chns)

        rectstart_right = rectstart_left + aioHead.rect1.datasz
        rect2_right = data[rectstart_right:rectstart_right + aioHead.rect2.datasz]
        right = np.frombuffer(rect2_right, dtype=np.uint8)
        right = right.reshape(aioHead.rect2.rows, aioHead.rect2.cols, aioHead.rect2.chns)

        ##
        dist1_left_start = headSize + aioHead.dist.datasz
        dist1_left = data[dist1_left_start:dist1_left_start + aioHead.dist1.datasz]
        dist1_image_left = np.frombuffer(dist1_left, dtype=np.uint8)
        dist1_image_left = dist1_image_left.reshape(aioHead.dist1.rows, aioHead.dist1.cols, aioHead.dist1.chns)

        dist2_right_start = dist1_left_start + aioHead.dist1.datasz
        dist2_left = data[dist2_right_start:dist2_right_start + aioHead.dist2.datasz]
        dist2_image_right = np.frombuffer(dist2_left, dtype=np.uint8)
        dist2_image_right = dist2_image_right.reshape(aioHead.dist2.rows, aioHead.dist2.cols, aioHead.dist2.chns)

    cloud = ""
    if aioHead.cloud.datasz != 0:
        cloud_start = headSize + aioHead.dist.datasz + aioHead.dist1.datasz + aioHead.dist2.datasz + aioHead.rect.datasz + aioHead.rect1.datasz \
                      + aioHead.rect2.datasz + aioHead.disp.datasz + aioHead.depth.datasz
        cloud_data = data[cloud_start:cloud_start + aioHead.cloud.datasz]
        cloud_numpy = np.frombuffer(cloud_data, dtype=np.float32)
        cloud = cloud_numpy.reshape(aioHead.cloud.rows, aioHead.cloud.cols, aioHead.cloud.chns)
    # BinoParamV = data[1204:2940]
    # intrinsic = BinoParamV[172:204]
    # trans = BinoParamV[204:236]

    BinoParamV = data[1204:2940]
    intrinsic = BinoParamV[204:236]
    trans = BinoParamV[236:260]

    intrinsic = np.frombuffer(intrinsic, dtype=np.float64)
    trans = np.frombuffer(trans, dtype=np.float64)

    disp_offset = headSize + aioHead.dist.datasz + aioHead.dist1.datasz + aioHead.dist2.datasz + aioHead.rect.datasz + aioHead.rect1.datasz + aioHead.rect2.datasz
    disp_value = data[disp_offset:disp_offset + aioHead.disp.datasz]
    dis = np.frombuffer(disp_value, dtype=np.float32)
    dis = dis.reshape(aioHead.disp.rows, aioHead.disp.cols)

    return left, right, intrinsic, trans, dis, dist1_image_left, dist2_image_right, cloud
