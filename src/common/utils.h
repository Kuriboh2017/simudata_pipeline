#ifndef screen_dirt_SRC_UTILS_H_
#define screen_dirt_SRC_UTILS_H_

#include <opencv2/opencv.hpp>
#include <GLES3/gl32.h>

#include <string>
#include <vector>

namespace utils
{
    double degreesToRadians(double degrees);

    cv::Matx33d GetRotation(double r, double p, double y);

    // Gets color format from the number of components.
    GLenum getColorFormat(int num_of_components);

    // Loads a 2D texture from a char array in memory
    unsigned int loadTexture(unsigned char *data, int width, int height, int num_of_components);

    // Loads a 2D texture from a file and output the width and height.
    unsigned int loadTexture(char const *path, int &width, int &height);

    // Loads a 2D texture from a file.
    unsigned int loadTexture(char const *path);

    // Reads a PFM file.
    std::vector<float> readPFMfile(const std::string &path, int &width, int &height, bool &grayscale, float &scalef);

    // Visualizes the image data.
    void visualizeImageData(const float *image_data, int width, int height, bool grayscale);

    // Writes the current render target to file
    void writeToFile(const std::string &file, int width, int height, int num_of_components);

    // Writes the buffer to a file
    void writeToFile(const std::string &file, int width, int height, int num_of_components, unsigned char *buffer);

    // Writes the buffer to a file
    void writePFMfile(const float *const image_data, int width, int height, const std::string &path, bool grayscale, float scalef);
} // namespace utils

#endif
