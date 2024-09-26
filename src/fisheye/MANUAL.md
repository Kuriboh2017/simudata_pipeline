# Fisheye 

Sample images are available [here](README.md).

## Overview
Since the noises are applied to the fisheye images in the real-world, we project our output to the round fisheye images to add noises. Specifically, we have the following procedures. 
1. Project the panorama image to the round fisheye images.
2. Secondly, we add noises on the fisheye images. 
3. Finally, we apply the same 1-fisheye-to-3-pinholes remapping function to the fisheye image to generate the final image, which makes the noises closer to the real-world image noises.

Two sets of executables are provided. One set of executables aims to process each image individually, while the other set of executables aims to process the images in batch across all three steps. Run each executable with `--help` to check the specific manual. For example,
```
$ run_fisheye.py --help
usage: run_fisheye.py [-h] -i INPUT_DIR [-f FISHEYE_DIR] -o OUTPUT_DIR [-r RECTIFY_CONFIG] [-use-r] [-k] [--blur]
                      [--blur-kernel-size BLUR_KERNEL_SIZE] [-l CALIBRATION_NOISE_LEVEL] [-roll ROLL] [-pitch PITCH]
                      [-yaw YAW]

Run fisheye rgba pipeline: panorama -> fisheye -> rectify

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory of the input images
  -f FISHEYE_DIR, --fisheye-dir FISHEYE_DIR
                        Directory of the fisheye images
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory of the output images
  -r RECTIFY_CONFIG, --rectify-config RECTIFY_CONFIG
                        path of the rectification config file
  -use-r, --use-right-intrinsics
                        use right intrinsics instead of left intrinsics
  -k, --keep-intermediate-images
                        Whether to keep the intermediate fisheye images
  --blur                Whether to add the gaussian blur effects
  --blur-kernel-size BLUR_KERNEL_SIZE
                        Gaussian blur kernel size
  -l CALIBRATION_NOISE_LEVEL, --calibration-noise-level CALIBRATION_NOISE_LEVEL
                        noise level of the camera intrinsic parameters
  -roll ROLL            extra roll angle in degrees
  -pitch PITCH          extra pitch angle in degrees
  -yaw YAW              extra yaw angle in degrees

```

## Individual executables

After the [installation](../../README.md#quickstart), you should have the following executables in your PATH.
1. Project a panorama image to two round fisheye images.
   * [panorama_to_fisheye.py](panorama_to_fisheye/panorama_to_fisheye.py)

2. Add noises like Gaussian blur
   * [add_blur.py](add_blur/add_blur.py)

3. Fisheye to 3-pinholes image
   * [fisheye_to_3pinholes.py](fisheye_to_3pinholes/fisheye_to_3pinholes.py)

4. Panorama to 3-pinholes image
   * [panorama_to_3pinholes.py](panorama_to_3pinholes/panorama_to_3pinholes.py)

Among them, 1,2,3 is used to generate the RGB scene images, while 4 is used to generate the ground-truth depth and segmentation-graymap images. Besides, the Gaussian blur is optionally applied to the RGB scene images, while the Gaussian blur is not applied to the depth and segmentation-graymap images.

## Batch processing executables

After the [installation](../../README.md#quickstart), you should also have the following executables in your PATH.
The executable below will run all steps (panorama -> fisheye -> 3-pinholes) together.
* [run_fisheye.py](../../batchfiles/run_fisheye.py)

The executable below will run the panorama to 3-pinholes step without the fisheye step.
* [run_panorama_to_3pinholes.py](../../batchfiles/run_panorama_to_3pinholes.py)

The executable below will run both executables above against a sim dataset.
* [run_fisheye_sim_dataset.py](../../batchfiles/run_fisheye_sim_dataset.py)

## Utils
To visualize the depth image (`.pfm` or `.npz`) in color, a utility executable is provided.
* [visualize_depth.py](../common/visualize_depth.py)

To visualize the segmentation-graymap image (`.png`), a utility executable is provided.
* [visualize_seg_graymap.py](../common/visualize_seg_graymap.py)
