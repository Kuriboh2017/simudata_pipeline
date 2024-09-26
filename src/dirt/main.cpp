#include "utils.h"
#include <EGL/egl.h>
#include <GLES3/gl32.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "arguments.h"

#include "shader.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {
static constexpr EGLint kConfigAttribs[] = {EGL_SURFACE_TYPE,
                                            EGL_PBUFFER_BIT,
                                            EGL_BLUE_SIZE,
                                            8,
                                            EGL_GREEN_SIZE,
                                            8,
                                            EGL_RED_SIZE,
                                            8,
                                            EGL_DEPTH_SIZE,
                                            8,
                                            EGL_RENDERABLE_TYPE,
                                            EGL_OPENGL_BIT,
                                            EGL_NONE};
} // namespace

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  const std::string input_image_path = FLAGS_input_image_path;
  if (input_image_path.empty()) {
    std::cerr << "Error: please specify the --input-image-path!\n";
    exit(1);
  }
  const std::string output_image_path = FLAGS_output_image_path;
  const float blur_intensity = static_cast<float>(FLAGS_blur_intensity);
  const int dirt_texture_offset_x = FLAGS_dirt_texture_offset_x;
  const int dirt_texture_offset_y = FLAGS_dirt_texture_offset_y;
  const float dirt_texture_rotation =
      static_cast<float>(FLAGS_dirt_texture_rotation);
  const float dirt_texture_scale = static_cast<float>(FLAGS_dirt_texture_scale);
  const float blur_kernel_size = static_cast<float>(FLAGS_blur_kernel_size);
  const float blur_pseudo_overexposure =
      static_cast<float>(FLAGS_blur_pseudo_overexposure);
  const float dirt_texture_red_scale =
      static_cast<float>(FLAGS_dirt_texture_red_scale);
  const float dirt_texture_green_scale =
      static_cast<float>(FLAGS_dirt_texture_green_scale);
  const float dirt_texture_blue_scale =
      static_cast<float>(FLAGS_dirt_texture_blue_scale);

  const int dirt_texture_id = FLAGS_dirt_texture_id;
  const float dirt_texture_ratio = static_cast<float>(FLAGS_dirt_texture_ratio);
  if (dirt_texture_id < 0 || dirt_texture_id > 15) {
    std::cerr << "Error: please specify a dirt texture id in between 0 and 15!";
    exit(1);
  }

  // 1. Initialize EGL
  EGLDisplay egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint major, minor;
  eglInitialize(egl_dpy, &major, &minor);
  std::cout << "Egl version: " << major << "." << minor << std::endl;

  // 2. Select an appropriate configuration
  EGLint num_configs;
  EGLConfig egl_cfg;
  eglChooseConfig(egl_dpy, kConfigAttribs, &egl_cfg, 1, &num_configs);

  // 3. Create a surface
  int src_width, src_height, src_num_of_components;
  unsigned char *original_img_data =
      stbi_load(input_image_path.c_str(), &src_width, &src_height,
                &src_num_of_components, 0);
  const EGLint pbuffer_attribs[] = {
      EGL_WIDTH, src_width, EGL_HEIGHT, src_height, EGL_NONE,
  };

  EGLSurface egl_surf =
      eglCreatePbufferSurface(egl_dpy, egl_cfg, pbuffer_attribs);

  // 4. Bind the API
  eglBindAPI(EGL_OPENGL_API);

  // 5. Create a context and make it current
  EGLContext egl_ctx = eglCreateContext(egl_dpy, egl_cfg, EGL_NO_CONTEXT, NULL);

  eglMakeCurrent(egl_dpy, egl_surf, egl_surf, egl_ctx);

  // From now on use your OpenGL context
  unsigned int original = utils::loadTexture(original_img_data, src_width, src_height,
                                      src_num_of_components);

  std::filesystem::path current_folder =
      std::filesystem::path(__FILE__).parent_path();
  std::string dirt_texture_filename =
      "screen_dirt_" + std::to_string(dirt_texture_id) + ".TGA";
  std::filesystem::path texture_path =
      current_folder / "textures" / dirt_texture_filename;
  unsigned int screen_effect = utils::loadTexture(texture_path.c_str());

  glClearColor(0.2f, 0.3f, 0.3f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  std::filesystem::path vertex_shader = current_folder / "dirt.vs";
  std::filesystem::path fragment_shader = current_folder / "dirt.fs";
  Shader ourShader(vertex_shader.c_str(), fragment_shader.c_str());

  // Configures the vertex attributes.
  float vertices[] = {
      // positions      // texture coords
      1.f,  1.f,  0.f, 1.f, 1.f, // top right
      1.f,  -1.f, 0.f, 1.f, 0.f, // bottom right
      -1.f, -1.f, 0.f, 0.f, 0.f, // bottom left
      -1.f, 1.f,  0.f, 0.f, 1.f  // top left
  };
  unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
  };
  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Set shader parameters.
  ourShader.Use();
  glUniform1i(glGetUniformLocation(ourShader.id(), "screen_effect"), 0);
  glUniform1i(glGetUniformLocation(ourShader.id(), "original"), 1);
  float resolution[] = {static_cast<float>(src_width),
                        static_cast<float>(src_height)};
  glUniform2fv(glGetUniformLocation(ourShader.id(), "resolution"), 1,
               resolution);

  glUniform1f(glGetUniformLocation(ourShader.id(), "blur_intensity"),
              blur_intensity);
  glUniform1f(glGetUniformLocation(ourShader.id(), "blur_pseudo_overexposure"),
              blur_pseudo_overexposure);
  glUniform1f(glGetUniformLocation(ourShader.id(), "blur_kernel_size"),
              blur_kernel_size);

  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_rotation"),
              dirt_texture_rotation);
  int dirt_texture_offset[] = {dirt_texture_offset_x, dirt_texture_offset_y};
  glUniform2iv(glGetUniformLocation(ourShader.id(), "dirt_texture_offset"), 1,
               dirt_texture_offset);
  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_scale"),
              dirt_texture_scale);
  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_ratio"),
              dirt_texture_ratio);
  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_red_scale"),
              dirt_texture_red_scale);
  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_green_scale"),
              dirt_texture_green_scale);
  glUniform1f(glGetUniformLocation(ourShader.id(), "dirt_texture_blue_scale"),
              dirt_texture_blue_scale);

  // Draw.
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, screen_effect);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, original);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  // Delete vertex buffer.
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);

  utils::writeToFile(output_image_path, src_width, src_height, src_num_of_components);
  std::cout << "Image output was generated at " << output_image_path
            << std::endl;

  // 6. Terminate EGL when finished
  eglTerminate(egl_dpy);
  return 0;
}
