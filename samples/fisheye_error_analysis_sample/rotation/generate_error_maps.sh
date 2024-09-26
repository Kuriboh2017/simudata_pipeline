#!/bin/bash

# Generate fisheye error maps for rotation angles noises
3pinholes_to_panorama.py -i 3pinholes.npz -o panorama.npz --float32
panorama_to_3pinholes.py -i panorama.npz -o0 rpy_0.05.npz --roll 0.05 --pitch 0.05 --yaw 0.05 --float32
check_fisheye_mapback.py -b 3pinholes.npz -e rpy_0.05.npz --error-visual-range 2.2

fisheye_to_panorama.py -i fisheye.png -o panorama.png
panorama_to_fisheye.py -i panorama.png -o0 rpy_0.0.png --roll 0.0 --pitch 0.0 --yaw 0.0
check_fisheye_mapback.py -b fisheye.png -e rpy_0.0.png --error-visual-range 12


# Generate fisheye error maps for focal lengths noises;
# scale the focal length 1.0015197568389 x 329 = 329.5
3pinholes_to_panorama.py -i 3pinholes.npz -o panorama.npz --float32
panorama_to_fisheye.py -i panorama.npz -o0 fisheye.npz --fxfy-scale 1.0015197568389 --recalculate --float32
fisheye_to_3pinholes.py -i fisheye.npz -o focal_length_0.5.npz --float32
check_fisheye_mapback.py -b 3pinholes.npz -e focal_length_0.5.npz --error-visual-range 3

# 1.0 baseline: keep the focal length unchanged
3pinholes_to_panorama.py -i 3pinholes.npz -o panorama.npz --float32
panorama_to_fisheye.py -i panorama.npz -o0 fisheye.npz --fxfy-scale 1.0 --recalculate --float32
fisheye_to_3pinholes.py -i fisheye.npz -o focal_length_1.0.npz --float32
check_fisheye_mapback.py -b 3pinholes.npz -e focal_length_1.0.npz --error-visual-range 1.0
