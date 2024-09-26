#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "arguments.h"
#include "utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

struct RectiSpec
{
    double row;
    double col;
    std::vector<double> param;
    double axis;
    RectiSpec(double row, double col, std::vector<double> param, double axis) : row(row), col(col), param(param), axis(axis) {}
};

std::string AppendFilename(const std::string &filename, const char *appendix)
{
    size_t pos = filename.find_last_of(".");
    std::string base = filename.substr(0, pos);
    std::string ext = filename.substr(pos);
    return base + "_" + std::string(appendix) + ext;
}

std::array<double, 2> ConvertXYZToPanoramaUV(const std::array<double, 3> &xyz)
{
    double u = 0.5 / M_PI * atan2(xyz[0], -xyz[1]) + 0.5;
    double v = atan2(sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]), xyz[2]) / M_PI;
    return {u, v};
}

std::pair<cv::Mat, cv::Mat> RemapCamera(const std::vector<double> &recti_param,
                                        const cv::Mat &rmat, double recti_row, double recti_col)
{
    cv::Mat irot = rmat.t();
    cv::Mat mapx = cv::Mat::zeros(recti_row, recti_col, CV_32F);
    cv::Mat mapy = cv::Mat::zeros(recti_row, recti_col, CV_32F);

    for (int v = 0; v < recti_row; v++)
    {
        for (int u = 0; u < recti_col; u++)
        {
            double xn = (u - recti_param[2]) / recti_param[0];
            double yn = (v - recti_param[3]) / recti_param[1];
            cv::Mat p3d = (cv::Mat_<double>(3, 1) << xn, yn, 1.0);

            cv::Mat p3d_new = irot * p3d;

            std::array<double, 2> uv = ConvertXYZToPanoramaUV(p3d_new);
            mapx.at<float>(v, u) = uv[0] * 2560;
            mapy.at<float>(v, u) = uv[1] * 1280;
        }
    }
    return {mapx, mapy};
}

std::pair<cv::Mat, cv::Mat> GenerateRemapTable(const std::vector<RectiSpec> &recti_specs,
                                               const cv::Matx33d &rot_noise, bool down)
{
    std::vector<cv::Mat> left_remap_x_list, left_remap_y_list;
    cv::Mat mat_noise(rot_noise);

    double pitch = down ? 90 : -90;
    cv::Mat up_or_down = (cv::Mat_<double>(3, 3) << std::cos(pitch * M_PI / 180.0), 0, std::sin(pitch * M_PI / 180.0),
                          0, 1, 0,
                          -std::sin(pitch * M_PI / 180.0), 0, std::cos(pitch * M_PI / 180.0));

    for (auto &rectispec : recti_specs)
    {
        double roll = rectispec.axis;
        cv::Mat rmat0 = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                         0, std::cos(roll * M_PI / 180.0), -std::sin(roll * M_PI / 180.0),
                         0, std::sin(roll * M_PI / 180.0), std::cos(roll * M_PI / 180.0));

        cv::Mat rmat1_rot = rmat0 * mat_noise * up_or_down;
        cv::Mat rmat2_rot = rmat0 * mat_noise * up_or_down;

        auto [left_x, left_y] = RemapCamera(rectispec.param, rmat1_rot, rectispec.row, rectispec.col);

        left_remap_x_list.push_back(left_x);
        left_remap_y_list.push_back(left_y);
    }

    cv::Mat left_remap_x, left_remap_y;
    cv::vconcat(left_remap_x_list, left_remap_x);
    cv::vconcat(left_remap_y_list, left_remap_y);
    return {left_remap_x, left_remap_y};
}

std::vector<RectiSpec> ReadYaml(const std::string &path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

    cv::FileNode multi_recti_params_node = fs["multiRectis"];
    std::vector<RectiSpec> multi_recti_params;

    for (cv::FileNodeIterator it = multi_recti_params_node.begin(); it != multi_recti_params_node.end(); ++it)
    {
        cv::FileNode n = *it;
        double row = (double)n[3];
        double col = (double)n[2];
        std::vector<double> param = {(double)n[4], (double)n[5], (double)n[6], (double)n[7]};
        double axis = (double)n[8];
        multi_recti_params.emplace_back(RectiSpec(row, col, param, axis));
    }

    return multi_recti_params;
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
        output_image_path0 = AppendFilename(input_image_path, "remap_directly_0");
    }
    if (output_image_path1.empty() || output_image_path1 == "None")
    {
        output_image_path1 = AppendFilename(input_image_path, "remap_directly_1");
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

    auto multispecs = ReadYaml(rectify_config);

    for (const auto &spec : multispecs)
    {
        std::cout << spec.row << " " << spec.col << " "
                  << " " << spec.axis << "; param: \n";
        for (const auto &param : spec.param)
        {
            std::cout << param << " ";
        }
        std::cout << "\n";
    }

    cv::Matx33d rot_noise = utils::GetRotation(roll, pitch, yaw);
    auto [down_x, down_y] = GenerateRemapTable(multispecs, rot_noise, true);
    auto [up_x, up_y] = GenerateRemapTable(multispecs, rot_noise, false);

    cv::Mat down_img, up_img;
    // Choose interpolation method based on whether depth image or not
    int operation = depth ? cv::INTER_NEAREST : cv::INTER_LINEAR;
    cv::remap(input_img, down_img, down_x, down_y, operation, cv::BORDER_REPLICATE);
    cv::remap(input_img, up_img, up_x, up_y, operation, cv::BORDER_REPLICATE);
    cv::flip(down_img, down_img, 0);
    cv::flip(up_img, up_img, 0);

    cv::imwrite(output_image_path0, down_img);
    cv::imwrite(output_image_path1, up_img);

    std::cout << "down_img shape = " << down_img.rows << ", " << down_img.cols << std::endl;

    if (FLAGS_visualize)
    {
        cv::imshow("Original", input_img);
        cv::imshow("Remapped", down_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}
