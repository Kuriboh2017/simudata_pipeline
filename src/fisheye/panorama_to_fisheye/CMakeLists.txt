
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(panorama_to_fisheye panorama_to_fisheye.cpp arguments.cpp )
add_dependencies(panorama_to_fisheye LensEffects::Common)
target_link_libraries(panorama_to_fisheye PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags ${OpenCV_LIBS})
install (TARGETS panorama_to_fisheye RUNTIME DESTINATION bin)

install(PROGRAMS
    generate_panorama_to_fisheye_table.py
    panorama_to_fisheye.py
DESTINATION bin)
