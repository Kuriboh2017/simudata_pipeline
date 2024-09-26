import pickle as pkl
from lz4.frame import compress, decompress
import numpy as np


def lz4_compress(remap_data, output_path):
    with open(output_path, "wb") as f_seg_out:
        seg_data = {
            'segmentation': compress(np.array(remap_data), compression_level=9),
            'segment_shape': remap_data.shape,
            'segment_dtype': remap_data.dtype
        }
        pkl.dump(seg_data, f_seg_out)


def lz4_decompress(input_path):
    with open(input_path, "rb") as f:
        raw_data = pkl.load(f)
    decompressed_data = decompress(raw_data)
    data_shape = decompressed_data['segment_shape']
    data_dtype = decompressed_data['segment_dtype']
    data = np.frombuffer(data['segmentation'], dtype=data_dtype)
    return data.reshape(data_shape)
