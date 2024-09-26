#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>

#include "arguments.h"
#include "remap.h"
#include "utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

std::string AppendFilename(const std::string &filename, const char *appendix)
{
    std::filesystem::path p(filename);
    return p.stem().string() + "_" + std::string(appendix) + p.extension().string();
}

cv::Point2f ConvertXYZToPanoramaUV(const cv::Point3f &xyz)
{
    float u = 0.5 / M_PI * atan2(xyz.x, -xyz.y) + 0.5;
    float v = atan2(sqrt(xyz.x * xyz.x + xyz.y * xyz.y), xyz.z) / M_PI;
    return cv::Point2f(u, v);
}

float UnprojectEucm(float eucm_alpha, float eucm_beta, float x, float y)
{
    float r2 = x * x + y * y;
    float beta_r2 = eucm_beta * r2;
    float term_inside_sqrt = 1 - (2 * eucm_alpha - 1) * beta_r2;
    if (term_inside_sqrt < 0)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float numerator = 1 - eucm_alpha * eucm_alpha * beta_r2;
    float denominator = eucm_alpha * sqrt(term_inside_sqrt) + 1 - eucm_alpha;
    return numerator / denominator;
}

std::tuple<cv::Mat, cv::Mat> GenerateRemapTable(int width, int height,
                                                const std::vector<float> &calib_param,
                                                const cv::Matx33f &rot_noise, bool down)
{
    float fx = calib_param[0];
    float fy = calib_param[1];
    float cx = calib_param[2];
    float cy = calib_param[3];
    float eucm_alpha = calib_param[4];
    float eucm_beta = calib_param[5];

    float pitch = down ? 90 : -90;

    cv::Matx33f rmat_pitch(cos(M_PI / 180 * pitch), 0, sin(M_PI / 180 * pitch),
                           0, 1, 0,
                           -sin(M_PI / 180 * pitch), 0, cos(M_PI / 180 * pitch));

    cv::Mat mapx = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat mapy = cv::Mat::zeros(height, width, CV_32F);

    for (int v = 0; v < height; ++v)
    {
        for (int u = 0; u < width; ++u)
        {
            float xn = (u - cx) / fx;
            float yn = (v - cy) / fy;
            float zn = UnprojectEucm(eucm_alpha, eucm_beta, xn, yn);

            if (std::isnan(zn))
            {
                mapx.at<float>(v, u) = std::numeric_limits<float>::quiet_NaN();
                mapy.at<float>(v, u) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            cv::Point3f normal(xn, yn, zn);
            normal = rot_noise.t() * normal;
            normal = rmat_pitch.t() * normal;
            cv::Point2f uv = ConvertXYZToPanoramaUV(normal);
            mapx.at<float>(v, u) = uv.x * 2560;
            mapy.at<float>(v, u) = uv.y * 1280;
        }
    }
    return std::make_tuple(mapx, mapy);
}

std::tuple<float, float, std::vector<float>> ReadYaml(const std::string &path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    float calibCols = fs["calibCols"];
    float calibRows = fs["calibRows"];
    cv::Mat left_intri;
    fs["calibParam1"] >> left_intri;

    std::vector<float> left_intri_vec;
    left_intri.convertTo(left_intri_vec, CV_32F);
    return std::make_tuple(calibCols, calibRows, left_intri_vec);
}

std::string GetCalibrationFileFolder()
{
    std::filesystem::path p(__FILE__);
    p = p.parent_path();
    p = std::filesystem::absolute(p / "../fisheye_to_3pinholes");
    return p;
}

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::string input_image_path = FLAGS_input_image_path;
    std::string output_image_path0 = FLAGS_output_image_path0;
    std::string output_image_path1 = FLAGS_output_image_path1;
    std::string rectify_config = FLAGS_rectify_config;
    const bool visualize = FLAGS_visualize;
    const bool depth = FLAGS_depth;
    const double calibration_noise_level = FLAGS_calibration_noise_level;
    const double roll = FLAGS_roll;
    const double pitch = FLAGS_pitch;
    const double yaw = FLAGS_yaw;
    if (input_image_path.empty())
    {
        std::cerr << "Error: please specify the --input-image-path!\n";
        exit(1);
    }
    if (output_image_path0.empty() || output_image_path0 == "None")
    {
        output_image_path0 = AppendFilename(input_image_path, "fisheye_0");
    }
    if (output_image_path1.empty() || output_image_path1 == "None")
    {
        output_image_path1 = AppendFilename(input_image_path, "fisheye_1");
    }
    if (rectify_config.empty() || rectify_config == "None")
    {
        rectify_config = GetCalibrationFileFolder() + "/rectification.yml";
    }
    std::cout << "Rectify_config path = " << rectify_config << "\n";
    cv::Mat input_img = cv::imread(input_image_path);
    std::cout << "Input_img shape = " << input_img.size() << "\n";

    if (input_image_path.find(".webp") != std::string::npos)
    {
        cv::rotate(input_img, input_img, cv::ROTATE_90_CLOCKWISE);
        cv::flip(input_img, input_img, 0);
    }

    float calibCols, calibRows;
    std::vector<float> left_intri;
    std::tie(calibCols, calibRows, left_intri) = ReadYaml(rectify_config);
    std::cout << "calibCols = " << calibCols << ", calibRows = " << calibRows << "\n";

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    cv::RNG rng(seed);
    for (auto &value : left_intri)
    {
        if (calibration_noise_level != 0.0)
        {
            value += rng.gaussian(value / 100.0 * calibration_noise_level);
        }
    }

    std::cout << "Intrinsic parameters = ";
    for (float value : left_intri)
    {
        std::cout << value << ", ";
    }
    std::cout << "\n";

    // Generate remap tables
    cv::Matx33d rot_noise = utils::GetRotation(roll, pitch, yaw);
    cv::Mat down_x, down_y, up_x, up_y;
    std::tie(down_x, down_y) = GenerateRemapTable(calibCols, calibRows, left_intri, rot_noise, true);
    std::tie(up_x, up_y) = GenerateRemapTable(calibCols, calibRows, left_intri, rot_noise, false);

    // Choose interpolation method based on whether depth image or not
    int operation = depth ? cv::INTER_NEAREST : cv::INTER_LINEAR;

    // Remap input image
    cv::Mat down_img, up_img;
    cv::remap(input_img, down_img, down_x, down_y, operation, cv::BORDER_REPLICATE);
    cv::remap(input_img, up_img, up_x, up_y, operation, cv::BORDER_REPLICATE);

    // Flip images
    cv::flip(down_img, down_img, 0);
    cv::flip(up_img, up_img, 0);

    // Save output images
    cv::imwrite(output_image_path0, down_img);
    cv::imwrite(output_image_path1, up_img);

    std::cout << "Output down_img shape = " << down_img.size() << ", path0 = " << output_image_path0 << std::endl;
    std::cout << "Output up_img shape = " << up_img.size() << ", path1 = " << output_image_path1 << std::endl;

    // Visualize if required
    if (visualize)
    {
        cv::imshow("Panorama", input_img);
        cv::imshow("Fisheye down", down_img);
        cv::imshow("Fisheye up", up_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
