add_executable(strong_flare main.cpp arguments.cpp )
add_dependencies(strong_flare LensEffects::Common)
target_link_libraries(strong_flare PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags)

install (TARGETS strong_flare RUNTIME DESTINATION bin)
