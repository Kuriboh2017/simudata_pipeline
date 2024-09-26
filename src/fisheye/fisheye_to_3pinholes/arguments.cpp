#include "arguments.h"

#include <gflags/gflags.h>

// Define the command-line flags.
DEFINE_string(input_image_path, "", "Path to the input image.");
DEFINE_string(output_image_path, "", "Path to the output image.");
DEFINE_string(rectify_config, "", "Path to the rectification configuration.");
DEFINE_double(calibration_noise_level, 0.0, "Level of calibration noise.");
DEFINE_bool(depth, false, "Whether or not processing depth image");
DEFINE_bool(visualize, false, "Enable or disable visualization.");
