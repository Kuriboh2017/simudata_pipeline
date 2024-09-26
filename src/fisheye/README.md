# Procedures to add noises to fisheye images

## Overview
Since the noises are applied to the fisheye images in the real-world, we project our output to the round fisheye images to add noises. Specifically, we have the following procedures. 
1. Project the panorama image to the round fisheye images.
2. Secondly, we add noises on the fisheye images. 
3. Finally, we apply the same 1-fisheye-to-3-pinholes remapping function to the fisheye image to generate the final image, which makes the noises closer to the real-world image noises.

Kudos to Dr. Zhenpeng Bian, the 1-fisheye-to-3-pinholes remapping is illustrated in the image below.
![diagram 1to3](../../samples/fisheye/benny_diagram_1to3.png)
![fisheye mesh remap](../../samples/fisheye/benny_fisheye_mesh_remap.png)

<img src="../../samples/fisheye/benny_raw_fish.png"  width="43%" height="43%">
<img src="../../samples/fisheye/benny_f2p.png"  width="40%" height="40%">

## Simulator output

Simulator outputs panorama images like below.
Kudos to Dr. Zhenpeng Bian for the regularly divided benchmark image below.

* Panorama scene image
![panorama scene image](../../samples/fisheye/scene.png)
![panorama scene image benchmark color](../../samples/fisheye/input_img.png)

<!-- * Panorama depth image
![panorama depth image](../../samples/fisheye/panorama_depth.png)

* Panorama segmentation image
![panorama segmentation image](../../samples/fisheye/segmentation.png) -->

## Converts a panorama image to two fisheye image.

* Fisheye scene images

<img src="../../samples/fisheye/scene_fisheye_0.png"  width="40%" height="40%">
<img src="../../samples/fisheye/input_img_fisheye_0.png"  width="40%" height="40%">

<!-- * Fisheye depth images

![depth down image](../../samples/fisheye/depth_bottom.png)
![depth up image](../../samples/fisheye/depth_up_color.png) -->

<!-- * Fisheye segmentation images

![segmentation down image](../../samples/fisheye/segmentation_down.png)
![segmentation up image](../../samples/fisheye/segmentation_up.png) -->

<!-- ## Adds noises on the fisheye image

* Fisheye scene noisy image

![noisy image](../../samples/fisheye/sample1_fisheye_0_gaussian_blurred.png) -->


## Converts/rectifies one noisy fisheye image to 3 pinhole images.

* Rectified original image

<img src="../../samples/fisheye/scene_fisheye_0_remapped.png"  width="40%" height="40%">
<img src="../../samples/fisheye/input_img_fisheye_0_remapped.png"  width="40%" height="40%">



<!-- * Rectified noisy image

![Rectified noisy image](../../samples/fisheye/sample1_fisheye_0_gaussian_blurred_remapped.png) -->

<!-- 
* Rectified depth down image

![Rectified depth image](../../samples/fisheye/depth_remapped.png)

* Rectified segmentation up image

![Rectified segmentation image](../../samples/fisheye/segmentation_up_remapped.png) -->
