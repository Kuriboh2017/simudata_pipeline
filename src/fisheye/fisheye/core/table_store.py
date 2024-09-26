from datetime import datetime
from pathlib import Path
import numpy as np
import os
import pickle
import uuid

from fisheye.core.types import CameraType, RemappingTableParams, RemappingTable
from fisheye.core.remapping_tables import (
    generate_panorama_to_fisheye_table,
    generate_panorama_to_3pinholes_table,
    generate_fisheye_to_3pinholes_table)

kRootPath = '/media/autel/sim_ssd/.remapping_tables'
kTablePath = os.path.join(kRootPath, 'tables')
kMetaName = 'remapping_table_meta'
kMetaPath = os.path.join(kRootPath, 'metas')


def load_table_meta():
    meta = {}
    for file in Path(kMetaPath).rglob(f'{kMetaName}*.pickle'):
        with open(file, 'rb') as f:
            meta.update(pickle.load(f))
    return meta


def _get_name(prefix, suffix):
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4()
    first_six = str(unique_id).replace('-', '')[:6]
    return f'{prefix}_{date_time_str}_{first_six}{suffix}'


def _save_remapping_table(remapping_table: RemappingTable):
    down_x, down_y, up_x, up_y = remapping_table
    folder = Path(kTablePath)
    folder.mkdir(parents=True, exist_ok=True)
    name = _get_name('remapping_table', '.npz')
    file_name = folder / name
    np.savez_compressed(file_name, down_x=down_x,
                        down_y=down_y, up_x=up_x, up_y=up_y)
    print(f'Generated table {file_name}')
    return file_name


def save_table_meta(meta):
    folder = Path(kMetaPath)
    folder.mkdir(parents=True, exist_ok=True)
    name = _get_name(kMetaName, '.pickle')
    saving_name = f'_saving_{name}'
    saving_file_name = folder / saving_name
    file_name = folder / name
    with open(saving_file_name, 'wb') as f:
        pickle.dump(meta, f)
    os.rename(saving_file_name, file_name)
    print(f'Updated meta {file_name}')


class TableStore:
    @staticmethod
    def get_remapping_table(params: RemappingTableParams) -> RemappingTable:
        meta = load_table_meta()
        if params not in meta:
            table = TableStore.generate_remapping_table(params)
            TableStore.save_remapping_table(params, table)
            return table
        table_path = meta[params]
        with np.load(table_path, allow_pickle=True) as data:
            down_x = data['down_x']
            down_y = data['down_y']
            up_x = data['up_x']
            up_y = data['up_y']
            param = params.dst
            assert down_x.shape == down_y.shape == (param.height, param.width), \
                f'Remapping table shape mismatch: down_x.shape {down_x.shape} != '\
                f'shapes ({param.height}, {param.width}) '
            return RemappingTable(down_x, down_y, up_x, up_y)

    @staticmethod
    def save_remapping_table(params: RemappingTableParams, remapping_table: RemappingTable):
        meta = load_table_meta()
        if params in meta:
            return
        table_path = _save_remapping_table(remapping_table)
        save_table_meta({params: table_path})

    @staticmethod
    def generate_remapping_table(params: RemappingTableParams) -> RemappingTable:
        if params.src.camera_type == CameraType.PANORAMA and \
                params.dst.camera_type == CameraType.FISHEYE:
            return generate_panorama_to_fisheye_table(params)
        elif params.src.camera_type == CameraType.PANORAMA and \
                params.dst.camera_type == CameraType.THREE_PINHOLES:
            return generate_panorama_to_3pinholes_table(params)
        elif params.src.camera_type == CameraType.FISHEYE and \
                params.dst.camera_type == CameraType.THREE_PINHOLES:
            return generate_fisheye_to_3pinholes_table(params)
        else:
            assert False, (
                f'Unsupported remapping table from {params.params.src.camera_type}!'
                f' to {params.params.dst.camera_type}. '
                'Please choose from panorama_to_fisheye, panorama_to_3pinholes, '
                'fisheye_to_3pinholes.'
            )
