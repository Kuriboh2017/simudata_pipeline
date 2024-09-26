from concurrent.futures import ProcessPoolExecutor
from types import Intrinsics, Rotation, ImageShapes, RemappingTableParams, RemappingTable
from typing import Callable, List
import copy


def _generate_params(original_params: RemappingTableParams, peak_ratio: float, num_of_steps: int):
    assert num_of_steps > 0, 'num_of_steps must be positive'
    list_of_params = []
    for i in range(num_of_steps + 1):
        ratio = 1. + i / num_of_steps * peak_ratio
        params = copy.deepcopy(original_params)
        params.intrinsics.fx *= ratio
        params.intrinsics.fy *= ratio
        list_of_params.append(params)
    return list_of_params


def _apply_func(func: Callable[[RemappingTableParams], RemappingTable], list_of_params: List[RemappingTableParams]):
    with ProcessPoolExecutor() as executor:
        futures = {params: executor.submit(func, params)
                   for params in list_of_params}
    return {params: future.result() for params, future in futures.items()}


def _main():
    image_shapes = ImageShapes(1280, 2560, 1120, 1120)
    intrinsics_left = Intrinsics(3.2934384416965571e+02, 3.2944972473240313e+02,
                                 559.5, 559.5,
                                 6.3360826858996278e-01, 1.1187501794190957e+00)
    intrinsics_right = Intrinsics(3.2945130811948769e+02, 3.2960781096670706e+02,
                                  559.5, 559.5,
                                  6.3484929545892321e-01, 1.1181950475528866e+00)
    rpy_noise = Rotation(0.0, 0.0, 0.0)
    params_right = RemappingTableParams(
        intrinsics_right, rpy_noise, image_shapes, 'panorama2fisheye')
    params_left = RemappingTableParams(
        intrinsics_left, rpy_noise, image_shapes, 'panorama2fisheye')

    set_of_params = set()
    set_of_params.update(_generate_params(params_right, 0.01, 260))
    set_of_params.update(_generate_params(params_right, 0.005, 260))
    set_of_params.update(_generate_params(params_left, -0.005, 260))

    meta = load_table_meta()
    remaining_set_of_params = [
        params for params in set_of_params if params not in meta]
    print(
        f'Requesting {len(set_of_params)} tables; Generating {len(remaining_set_of_params)} tables...')
    if remaining_set_of_params:
        results = _apply_func(_generate_p2f_remap_tables,
                              remaining_set_of_params)
        additional_meta = {params: _save_remapping_table(table)
                           for params, table in results.items()}
        save_table_meta(additional_meta)


if __name__ == '__main__':
    _main()
