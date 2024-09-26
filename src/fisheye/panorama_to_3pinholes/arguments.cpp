#include "arguments.h"

// Define your gflags
DEFINE_string(input_image_path, "", "Path of the input image");
DEFINE_string(output_image_path0, "", "Path 0 of the output image");
DEFINE_string(output_image_path1, "", "Path 1 of the output image");
DEFINE_string(rectify_config, "", "Path of the rectification config file");
DEFINE_bool(visualize, false, "Visualize the rectified image");
DEFINE_bool(depth, false, "Whether or not processing depth image");
DEFINE_double(roll, 0.0, "Extra roll degrees of the camera");
DEFINE_double(pitch, 0.0, "Extra pitch degrees of the camera");
DEFINE_double(yaw, 0.0, "Extra yaw degrees of the camera");
