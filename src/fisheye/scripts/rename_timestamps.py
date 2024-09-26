import itertools
from pathlib import Path
import numpy as np

old_timestamps = [1688987906395]
new_timestamps = [1688987906395]
num_of_steps = 521
for _ in range(num_of_steps):
    old_timestamps.append(old_timestamps[-1] + 20)
    new_timestamps.append(new_timestamps[-1] + 100)

old_timestamps.sort(reverse=True)
new_timestamps.sort(reverse=True)

# suffix = ['_down.webp', '_up.webp']
suffix = ['_down.npz', '_up.npz']

for s, i in itertools.product(suffix, range(num_of_steps)):
    old_file = f'{old_timestamps[i]}{s}'
    new_file = f'{new_timestamps[i]}{s}'
    assert old_file != new_file, f'{old_file} == {new_file}'
    assert Path(old_file).exists(), f'Error! {old_file} does not exist'
    assert not Path(new_file).exists(), f'Error! {new_file} already exist'
    Path(old_file).rename(new_file)
