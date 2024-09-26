cmake_minimum_required (VERSION 3.0)
cmake_policy(VERSION 3.0)
project(lens_effects)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(gflags REQUIRED)

include_directories(thirdparty/stb)
add_subdirectory(src)
add_subdirectory(batchfiles)
