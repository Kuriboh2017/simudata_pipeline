# Fisheye Postprocess Pipeline 

The pipeline is actively updated upon the requests from the perception and the calibration team.

The current procedures are generally as follows:
1. Use a proper `settings.json` to run our customized Unreal Engine 5 / AirSim to generate the raw RGB/Depth/Segmentation images.

2. Rotate and the compress the generated images. There are two options to do that:

   2.1 Use `perception_preprocess_pipeline` ([link](http://gitlab-jh.private-cloud.autelrobotics.com/autonomy/autonomy-cloud/data-platform/datasets-preprocess-management/perception_preprocess_pipeline)) to rotate and compress the images. Please consult Jianhua or Xuanquan about the procedures.

   2.2 Use the `post_sim_process.py` ([link](../../batchfiles/post_sim_process.py)) to rotate and compress the fisheye images.

3. Run the fisheye pipelines to generate the fisheye images. In this step, the requirements are often different for different tasks. So a new batch file is often created for each task. Nevertheless, the batch file is usually consistent of calling the following individual conversion scripts:

   3.1 Convert the panorama image to two fisheye images. (Refer to [link](panorama_to_fisheye/panorama_to_fisheye.py) for the conversion for an individual panorama image). It is optional to add calibration rotation noise in this step.

   3.2 Add ISP effects (e.g., blur, noise, etc.) to the fisheye images. (Refer to [link](add_effects/) for the conversion for an individual fisheye image).

   3.3 Perform the fisheye undistortion to 3pinholes. (Refer to [link](fisheye_to_3pinholes/fisheye_to_3pinholes.py) for the conversion for an individual 3pinholes image).


   3.4 Above 3.1-3.3 are for RGB images. For the segmentation and depth images, we convert the panorama images to 3pinholes directly without the fisheye step to get the most accurate segmentation and depth images. The `panorama_to_3pinholes.py` ([link](panorama_to_3pinholes/panorama_to_3pinholes.py)) can do this job.

4. After the conversion, due to backward compatibility, the `group1/cam1_0` stores the front camera, and the `group1/cam1_1` stores the back camera. Besides, image names use `_up` and `_down` suffix to represent up and down fisheyes. To make the naming consistent with the real fisheye cameras, we need to make `group0` the down fisheye, the `group1` the up fisheye, and remove the `_up` and `_down` suffixes. Furthermore, we need to rotate the up fisheye 180 degrees to have a consistent left/right output folder structure with the real fisheye cameras. After the conversion, `camX_0` is the `left` camera, `camX_1` is the `right` camera. For this rename procedure, the `post_fisheye_process.py` ([link](../../batchfiles/post_fisheye_process.py)) can do this job. In addition, for the new 3cubes output folder structure, we need to use the `cube3_post_fisheye_process.py` ([link](../../batchfiles/cube3_post_fisheye_process.py)) to do the job.


Above is the general procedure of the fisheye pipeline. But this pipeline is actively updated. So it is possible the image format or folder structure has minor changes. Please use it with caution. It is better to check each step of the pipeline to make sure the output is correct. Some check scripts are provided in [link1](scripts/check_img_similiarity_after_warp.py), [link2](scripts/check_stereo_feature.py), and [link3](scripts/check_stereo_warp.py).



