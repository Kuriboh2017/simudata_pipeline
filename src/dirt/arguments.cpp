#include "arguments.h"

DEFINE_string(input_image_path, "", "Path to an input image");
DEFINE_string(output_image_path, "output.png", "Path to the output image");

DEFINE_double(blur_intensity, 1.0, "Intensity of the screen dirt");
DEFINE_double(blur_kernel_size, 9.0, "Blur range per pixel");
DEFINE_double(blur_pseudo_overexposure, 3.0,
              "Fake the excessive exposure in the blur algorithm");

// Dirt textures are available in folder:
// "simulation/screen_effects/dirt/src/textures/".
DEFINE_int32(dirt_texture_id, 3, "Specify the dirt texture to use.");
DEFINE_double(dirt_texture_ratio, 0.3,
              "Ratio of the dirt texture overlay in the final image");
DEFINE_int32(dirt_texture_offset_x, 0.0, "Offset X of the dirty texture");
DEFINE_int32(dirt_texture_offset_y, 0.0, "Offset Y of the dirty texture");
DEFINE_double(dirt_texture_rotation, 0.0,
              "Rotation ange of the dirty texture in radians");
DEFINE_double(dirt_texture_scale, 1.0, "Scale of the dirty texture");
DEFINE_double(dirt_texture_red_scale, 1.0,
              "Scale the red component of the blur color");
DEFINE_double(dirt_texture_green_scale, 1.0,
              "Scale the green component of the blur color");
DEFINE_double(dirt_texture_blue_scale, 1.0,
              "Scale the blue component of the blur color");
