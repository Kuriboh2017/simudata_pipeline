#include "arguments.h"

DEFINE_string(input_image_path, "", "Path to an input image");
DEFINE_string(output_image_path, "output.png", "Path to the output image");

DEFINE_double(sun_size, 1.0, "Size of the generated Sun");

// The location of the generated Sun in the image. The top left of the image is
// (-1,-1), and the bottom right of the image is (1,1).
DEFINE_double(sun_location_x, -0.6, "Sun location x between -1.0 and 1.0");
DEFINE_double(sun_location_y, -0.6, "Sun location y between -1.0 and 1.0");

DEFINE_double(diffraction_spikes_intensity, 1.0,
              "Overall intensity of the diffraction spikes around the Sun");
DEFINE_double(lens_flares_intensity, 1.0,
              "Overall intensity of the lens flares");
DEFINE_double(lens_flares_red_scale, 1.4,
              "Scale the red component of the lens flares");
DEFINE_double(lens_flares_green_scale, 1.2,
              "Scale the green component of the lens flares");
DEFINE_double(lens_flares_blue_scale, 1.0,
              "Scale the blue component of the lens flares");