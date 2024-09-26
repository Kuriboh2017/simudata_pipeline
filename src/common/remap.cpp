#include "remap.h"

#include <limits>

namespace utils
{

    void Remap4(const std::vector<float> &remap_table,
                int src_width, int src_height, const uint8_t *src_img,
                int dst_width, int dst_height, uint8_t *dst_img)
    {
        assert(remap_table.size() == dst_width * dst_height * 2);

        for (int r = 0; r < dst_height; ++r)
        {
            for (int c = 0; c < dst_width; ++c)
            {
                const int dst_index = r * dst_width + c;
                const float u = remap_table[dst_index * 2 + 0];
                const float v = remap_table[dst_index * 2 + 1];
                if (std::isnan(u) || u <= 0 || u >= 1 || std::isnan(v) || v <= 0 || v >= 1)
                {
                    dst_img[4 * dst_index + 0] = dst_img[4 * dst_index + 1] = dst_img[4 * dst_index + 2] = 0;
                    dst_img[4 * dst_index + 3] = 255;
                    continue;
                }
                // Bilinear sampling
                const float src_r_float = v * src_height;
                const float src_c_float = u * src_width;
                const float src_r_float_floor = std::floor(src_r_float);
                const float src_c_float_floor = std::floor(src_c_float);
                const float src_r_frac = src_r_float - src_r_float_floor;
                const float src_c_frac = src_c_float - src_c_float_floor;
                const int src_r = static_cast<int>(src_r_float_floor);
                const int src_c = static_cast<int>(src_c_float_floor);
                const int src_r_1 = std::min(src_r + 1, src_height - 1);
                const int src_c_1 = std::min(src_c + 1, src_width - 1);
                const int src_index_00 = src_r * src_width + src_c;
                const int src_index_01 = src_r * src_width + src_c_1;
                const int src_index_10 = src_r_1 * src_width + src_c;
                const int src_index_11 = src_r_1 * src_width + src_c_1;
                for (int i = 0; i < 4; ++i)
                {
                    const float ch = std::lerp(std::lerp(static_cast<float>(src_img[4 * src_index_00 + i]) / 255.f,
                                                         static_cast<float>(src_img[4 * src_index_01 + i]) / 255.f,
                                                         src_c_frac),
                                               std::lerp(static_cast<float>(src_img[4 * src_index_10 + i]) / 255.f,
                                                         static_cast<float>(src_img[4 * src_index_11 + i]) / 255.f,
                                                         src_c_frac),
                                               src_r_frac);
                    dst_img[4 * dst_index + i] = static_cast<uint8_t>(std::min(ch, 1.f) * 255.f);
                }
            }
        }
    }

    void RemapDepthImage(const std::vector<float> &remap_table,
                         int src_width, int src_height, const float *src_img,
                         int dst_width, int dst_height, float *dst_img)
    {
        assert(remap_table.size() == dst_width * dst_height * 2);

        for (int r = 0; r < dst_height; ++r)
        {
            for (int c = 0; c < dst_width; ++c)
            {
                const int dst_index = r * dst_width + c;
                const float u = remap_table[dst_index * 2 + 0];
                const float v = remap_table[dst_index * 2 + 1];
                if (std::isnan(u) || u <= 0 || u >= 1 || std::isnan(v) || v <= 0 || v >= 1)
                {
                    dst_img[dst_index] = std::numeric_limits<float>::max();
                    continue;
                }
                // Bilinear sampling
                const float src_r_float = v * src_height;
                const float src_c_float = u * src_width;
                const float src_r_float_floor = std::floor(src_r_float);
                const float src_c_float_floor = std::floor(src_c_float);
                const float src_r_frac = src_r_float - src_r_float_floor;
                const float src_c_frac = src_c_float - src_c_float_floor;
                const int src_r = static_cast<int>(src_r_float_floor);
                const int src_c = static_cast<int>(src_c_float_floor);
                const int src_r_1 = std::min(src_r + 1, src_height - 1);
                const int src_c_1 = std::min(src_c + 1, src_width - 1);
                const int src_index_00 = src_r * src_width + src_c;
                const int src_index_01 = src_r * src_width + src_c_1;
                const int src_index_10 = src_r_1 * src_width + src_c;
                const int src_index_11 = src_r_1 * src_width + src_c_1;
                dst_img[dst_index] = std::min(std::min(static_cast<float>(src_img[src_index_00]),
                                                       static_cast<float>(src_img[src_index_01])),
                                              std::min(static_cast<float>(src_img[src_index_10]),
                                                       static_cast<float>(src_img[src_index_11])));
            }
        }
    }

} // namespace utils
