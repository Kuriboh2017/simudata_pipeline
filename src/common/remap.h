#ifndef LENS_EFFECTS_SRC_COMMON_REMAP_H_
#define LENS_EFFECTS_SRC_COMMON_REMAP_H_

#include <stdint.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cassert>
#include <vector>

namespace utils
{
    constexpr double RadiansToDegrees(double radians)
    {
        return static_cast<double>(radians * 180.0 / M_PIl);
    }
    constexpr float RadiansToDegrees(float radians)
    {
        return static_cast<float>(radians * 180.0f / M_PI);
    }

    static constexpr double DegreesToRadians(double degrees)
    {
        return static_cast<double>(M_PIl * degrees / 180.0);
    }
    static constexpr float DegreesToRadians(float degrees)
    {
        return static_cast<float>(M_PI * degrees / 180.0f);
    }

    void Remap4(const std::vector<float> &remap_table,
                int src_width, int src_height, const uint8_t *src_img,
                int dst_width, int dst_height, uint8_t *dst_img);

    void RemapDepthImage(const std::vector<float> &remap_table,
                         int src_width, int src_height, const float *src_img,
                         int dst_width, int dst_height, float *dst_img);

} // namespace utils

#endif // LENS_EFFECTS_SRC_COMMON_REMAP_H_
