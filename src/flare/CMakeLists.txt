add_executable(lens_flare main.cpp arguments.cpp )
add_dependencies(lens_flare LensEffects::Common)
target_link_libraries(lens_flare PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags)

install (TARGETS lens_flare RUNTIME DESTINATION bin)
