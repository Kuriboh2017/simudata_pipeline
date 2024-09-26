# Code in this file was authored by Dr. Zhenyou Dai.
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fisheye_to_3pinholes fisheye_to_3pinholes.cpp arguments.cpp )
add_dependencies(fisheye_to_3pinholes LensEffects::Common)
target_link_libraries(fisheye_to_3pinholes PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags ${OpenCV_LIBS})
install (TARGETS fisheye_to_3pinholes RUNTIME DESTINATION bin)

install(PROGRAMS
    fisheye_to_3pinholes.py
    generate_fisheye_to_3pinholes_table.py
    rectify_depth.py
    rectification.yml
    rectification_old.yml
    rectification_new.yml
DESTINATION bin)
