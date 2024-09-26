
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})



# target_link_libraries(YourProjectName )

add_library(lens_effects_common shader.cpp utils.cpp remap.cpp)
target_link_libraries(lens_effects_common PUBLIC OpenGL::OpenGL OpenGL::EGL ${OpenCV_LIBS})

add_library(LensEffects::Common ALIAS lens_effects_common)

install(PROGRAMS
    visualize_depth.py
    visualize_seg_graymap.py
DESTINATION bin)
