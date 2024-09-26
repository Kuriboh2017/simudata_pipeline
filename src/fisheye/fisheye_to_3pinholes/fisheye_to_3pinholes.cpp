#include <chrono>
#include <cmath>
#include <filesystem>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "arguments.h"
#include "utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace
{
    namespace fs = std::filesystem;

    struct RectiSpec
    {
        int row;
        int col;
        std::vector<double> param;
        double axis;

        RectiSpec(int row, int col, std::vector<double> param, double axis)
            : row(row), col(col), param(std::move(param)), axis(axis) {}
    };

    std::string append_filename(const std::string &filename, const std::string &appendix = "remapped")
    {
        fs::path p(filename);
        return (p.stem().string() + "_" + appendix + p.extension().string());
    }

    std::pair<cv::Mat, cv::Mat> remap_camera(const std::vector<double> &calib_param, const std::vector<double> &recti_param,
                                             const cv::Mat &rmat, int recti_row, int recti_col)
    {
        double fx = calib_param[0];
        double fy = calib_param[1];
        double cx = calib_param[2];
        double cy = calib_param[3];

        double eucm_alpha = calib_param[4];
        double eucm_beta = calib_param[5];

        cv::Mat irot = rmat.t();

        cv::Mat mapx = cv::Mat::zeros(recti_row, recti_col, CV_32F);
        cv::Mat mapy = cv::Mat::zeros(recti_row, recti_col, CV_32F);

        for (int v = 0; v < recti_row; v++)
        {
            for (int u = 0; u < recti_col; u++)
            {
                double xn = (u - recti_param[2]) / recti_param[0];
                double yn = (v - recti_param[3]) / recti_param[1];
                cv::Mat p3d = (cv::Mat_<double>(3, 1) << xn, yn, 1.);
                cv::Mat p3d_new = irot * p3d;

                double x = p3d_new.at<double>(0, 0);
                double y = p3d_new.at<double>(1, 0);
                double z = p3d_new.at<double>(2, 0);

                double r2 = x * x + y * y;
                double rho2 = eucm_beta * r2 + z * z;
                double rho = std::sqrt(rho2);
                double norm = eucm_alpha * rho + (1.0 - eucm_alpha) * z;
                double mx = x / norm;
                double my = y / norm;

                mapx.at<float>(v, u) = fx * mx + cx;
                mapy.at<float>(v, u) = fy * my + cy;
            }
        }

        return {mapx, mapy};
    }

    std::tuple<cv::Mat, cv::Mat> generate_remap_table(const std::vector<double> &left_intri, const std::vector<RectiSpec> &recti_specs)
    {
        cv::Mat rmat1 = cv::Mat::eye(3, 3, CV_64F);

        std::vector<cv::Mat> left_remap_x_list, left_remap_y_list;

        for (const auto &rectispec : recti_specs)
        {
            cv::Mat rmat0 = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                             0, std::cos(utils::degreesToRadians(rectispec.axis)), -std::sin(utils::degreesToRadians(rectispec.axis)),
                             0, std::sin(utils::degreesToRadians(rectispec.axis)), std::cos(utils::degreesToRadians(rectispec.axis)));
            cv::Mat rmat1_rot = rmat0 * rmat1;

            auto [left_x, left_y] = remap_camera(left_intri, rectispec.param, rmat1_rot, rectispec.row, rectispec.col);

            left_remap_x_list.push_back(left_x);
            left_remap_y_list.push_back(left_y);
        }

        // concatenate along y axis
        cv::Mat left_remap_x, left_remap_y;
        cv::vconcat(left_remap_x_list, left_remap_x);
        cv::vconcat(left_remap_y_list, left_remap_y);
        return {left_remap_x, left_remap_y};
    }

    void read_yaml(const std::string &path, std::vector<double> &left_intri, std::vector<RectiSpec> &multi_recti_params)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        cv::Mat left_intri_mat;
        fs["calibParam1"] >> left_intri_mat;
        left_intri = left_intri_mat;

        cv::FileNode multi_recti_params_node = fs["multiRectis"];
        for (cv::FileNodeIterator it = multi_recti_params_node.begin(); it != multi_recti_params_node.end(); ++it)
        {
            std::vector<double> param = {(*it)[4].real(), (*it)[5].real(), (*it)[6].real(), (*it)[7].real()};
            multi_recti_params.emplace_back((*it)[3].real(), (*it)[2].real(), param, (*it)[8].real());
        }
    }

    std::string GetCalibrationFileFolder()
    {
        std::filesystem::path p(__FILE__);
        p = p.parent_path();
        p = std::filesystem::absolute(p);
        return p;
    }
} // namespace

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string input_image_path = FLAGS_input_image_path;
    if (input_image_path.empty())
    {
        std::cerr << "Error: input image path is empty" << std::endl;
        return -1;
    }
    std::string output_image_path = FLAGS_output_image_path;
    std::string rectify_config = FLAGS_rectify_config;
    double calibration_noise_level = FLAGS_calibration_noise_level;
    bool depth = FLAGS_depth;

    if (output_image_path.empty())
    {
        output_image_path = append_filename(input_image_path);
    }

    cv::Mat input_img = cv::imread(input_image_path);
    std::cout << "input_img shape = " << input_img.rows << " x " << input_img.cols << "\n";

    if (input_img.rows != 1120 || input_img.cols != 1120)
    {
        std::cerr << "Error: input image dimension is not 1120x1120" << std::endl;
        return -1;
    }

    if (calibration_noise_level < 0.0)
    {
        std::cerr << "Error: calibration noise level must be non-negative" << std::endl;
        return -1;
    }

    if (rectify_config.empty() || rectify_config == "None")
    {
        rectify_config = GetCalibrationFileFolder() + "/rectification.yml";
    }

    if (!std::filesystem::exists(rectify_config))
    {
        std::cerr << "Error: rectification config file " << rectify_config << " does not exist" << std::endl;
        return -1;
    }

    std::vector<double> left_intri;
    std::vector<RectiSpec> multi_recti_params;
    read_yaml(rectify_config, left_intri, multi_recti_params);

    // Add noise to the left intrinsic parameters.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    cv::RNG rng(seed);
    for (int i = 0; i < 6; ++i)
    {
        if (calibration_noise_level != 0.0)
        {
            left_intri[i] += rng.gaussian(left_intri[i] / 100.0 * calibration_noise_level);
        }
    }

    std::cout << "left intrinsics = ";
    for (int i = 0; i < 6; ++i)
    {
        std::cout << left_intri[i] << " ";
    }
    std::cout << "\n";

    for (const auto &spec : multi_recti_params)
    {
        std::cout << spec.row << " " << spec.col << " "
                  << " " << spec.axis << "; param: \n";
        for (const auto &param : spec.param)
        {
            std::cout << param << " ";
        }
        std::cout << "\n";
    }

    auto [lx, ly] = generate_remap_table(left_intri, multi_recti_params);

    // Choose interpolation method based on whether depth image or not
    int operation = depth ? cv::INTER_NEAREST : cv::INTER_LINEAR;

    cv::Mat output_img;
    cv::remap(input_img, output_img, lx, ly, operation);

    cv::imwrite(output_image_path, output_img);

    std::cout << "output_img shape = " << output_img.rows << " x " << output_img.cols << "\n";

    if (FLAGS_visualize)
    {
        cv::imshow("Original", input_img);
        cv::imshow("Remapped", output_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
