

#include <gflags/gflags.h>

DECLARE_string(input_image_path);
DECLARE_string(output_image_path);

DECLARE_double(blur_intensity);
DECLARE_double(blur_kernel_size);
DECLARE_double(blur_pseudo_overexposure);

DECLARE_int32(dirt_texture_id);
DECLARE_double(dirt_texture_ratio);
DECLARE_int32(dirt_texture_offset_x);
DECLARE_int32(dirt_texture_offset_y);
DECLARE_double(dirt_texture_rotation);
DECLARE_double(dirt_texture_scale);
DECLARE_double(dirt_texture_red_scale);
DECLARE_double(dirt_texture_green_scale);
DECLARE_double(dirt_texture_blue_scale);
