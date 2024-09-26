
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(panorama_to_3pinholes panorama_to_3pinholes.cpp arguments.cpp )
add_dependencies(panorama_to_3pinholes LensEffects::Common)
target_link_libraries(panorama_to_3pinholes PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags ${OpenCV_LIBS})
install (TARGETS panorama_to_3pinholes RUNTIME DESTINATION bin)

install(PROGRAMS
    generate_panorama_to_3pinholes_table.py
    panorama_to_3pinholes.py
DESTINATION bin)
