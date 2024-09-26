#include "utils.h"

#include <cmath>

#include <stb_image.h>
#include <stb_image_write.h>
#include <EGL/egl.h>
#include <GLES3/gl32.h>

#include <fstream>
#include <iostream>

namespace utils
{
    double degreesToRadians(double degrees)
    {
        return degrees * M_PI / 180.0;
    }

    cv::Matx33d GetRotation(double r, double p, double y)
    {
        const double roll = utils::degreesToRadians(r);
        const double pitch = utils::degreesToRadians(p);
        const double yaw = utils::degreesToRadians(y);
        cv::Matx33d rotationMatrix(
            std::cos(yaw) * std::cos(pitch),
            std::cos(yaw) * std::sin(pitch) * std::sin(roll) - std::sin(yaw) * std::cos(roll),
            std::cos(yaw) * std::sin(pitch) * std::cos(roll) + std::sin(yaw) * std::sin(roll),
            std::sin(yaw) * std::cos(pitch),
            std::sin(yaw) * std::sin(pitch) * std::sin(roll) + std::cos(yaw) * std::cos(roll),
            std::sin(yaw) * std::sin(pitch) * std::cos(roll) - std::cos(yaw) * std::sin(roll),
            -std::sin(pitch),
            std::cos(pitch) * std::sin(roll),
            std::cos(pitch) * std::cos(roll));

        return rotationMatrix;
    }

    GLenum getColorFormat(int num_of_components)
    {
        GLenum format;
        switch (num_of_components)
        {
        case 1:
        {
            format = GL_RED;
            break;
        }
        case 3:
        {
            format = GL_RGB;
            break;
        }
        case 4:
        {
            format = GL_RGBA;
            break;
        }
        default:
        {
            std::cerr << "Error: Unsupported num of components [" << num_of_components << "]!" << std::endl;
            break;
        }
        }
        return format;
    }

    unsigned int loadTexture(unsigned char *data, int width, int height, int num_of_components)
    {
        unsigned int textureID;
        glGenTextures(1, &textureID);

        if (data)
        {
            GLenum format = getColorFormat(num_of_components);

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            stbi_image_free(data);
        }
        else
        {
            std::cerr << "Error: Texture failed to load!" << std::endl;
            stbi_image_free(data);
        }

        return textureID;
    }

    unsigned int loadTexture(char const *path, int &width, int &height)
    {
        int nrComponents;
        unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
        return loadTexture(data, width, height, nrComponents);
    }

    unsigned int loadTexture(char const *path)
    {
        int width, height;
        return loadTexture(path, width, height);
    }

    std::vector<float> readPFMfile(const std::string &path, int &width, int &height, bool &grayscale, float &scalef)
    {
        std::ifstream file(path.c_str(), std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open file for reading.");
        }

        // Read the type of the file (bands)
        std::string bands;
        std::getline(file, bands);
        grayscale = (bands == "Pf");

        // Read the width and height
        file >> width >> height;

        // Read the scale factor
        file >> scalef;
        if (scalef < 0)
        {
            // If scale factor is negative, the file uses little endian
            scalef = -scalef;
        }

        // Skip the newline character
        file.ignore(1);

        // Create a buffer for the image data
        const int num_pixels = width * height * (grayscale ? 1 : 3);
        std::vector<float> image_data(num_pixels);

        // Read the image data from the file
        file.read(reinterpret_cast<char *>(&image_data[0]), num_pixels * sizeof(float));

        // Close the file
        file.close();

        return image_data;
    }

    void visualizeImageData(const float *image_data, int width, int height, bool grayscale)
    {
        // Convert to cv::Mat
        cv::Mat img;
        if (grayscale)
        {
            img = cv::Mat(height, width, CV_32F, const_cast<float *>(image_data));
        }
        else
        {
            img = cv::Mat(height, width, CV_32FC3, const_cast<float *>(image_data));
        }

        img.convertTo(img, CV_8U);

        // Display the image
        cv::namedWindow("Image", cv::WINDOW_NORMAL);
        cv::imshow("Image", img);

        // Wait for a key press before closing the window
        cv::waitKey(0);
    }

    void writeToFile(const std::string &file, int width, int height, int num_of_components)
    {
        GLenum format = getColorFormat(num_of_components);
        // Read pixel data and write to a file
        EGLint stride = num_of_components * width;
        stride += (stride % 4) ? (4 - stride % 4) : 0;
        EGLint buffer_size = stride * height;
        std::vector<char> buffer(buffer_size);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width, height, format, GL_UNSIGNED_BYTE, buffer.data());
        // stbi_flip_vertically_on_write(true);
        stbi_write_png(file.c_str(), width, height, num_of_components, buffer.data(), stride);
    }

    void writeToFile(const std::string &file, int width, int height, int num_of_components, unsigned char *buffer)
    {
        int stride = num_of_components * width;
        stride += (stride % 4) ? (4 - stride % 4) : 0;
        stbi_write_png(file.c_str(), width, height, num_of_components, buffer, stride);
    }

    void writePFMfile(const float *const image_data, int width, int height, const std::string &path, bool grayscale, float scalef)
    {
        std::ofstream file(path.c_str(), std::ios::binary);

        const std::string bands = grayscale ? "Pf" : "PF"; // grayscale or RGB

        // sign of scalefact indicates endianness, see pfm specs
        constexpr const bool kIsLittleEndian = true;
        if (kIsLittleEndian)
            scalef = -scalef;

        // insert header information
        file << bands << "\n";
        file << width << " ";
        file << height << "\n";
        file << scalef << "\n";

        const int data_size = sizeof(float) * width * height * (grayscale ? 1 : 3);
        file.write(reinterpret_cast<const char *>(image_data), data_size);

        file.close();
    }

} // namespace utils