
add_executable(egl_hello_world egl_hello_world.cpp)
target_link_libraries(egl_hello_world PRIVATE OpenGL::OpenGL OpenGL::EGL)

add_executable(screen_dirt main.cpp arguments.cpp)
add_dependencies(screen_dirt LensEffects::Common)
target_link_libraries(screen_dirt PRIVATE LensEffects::Common OpenGL::OpenGL OpenGL::EGL gflags)

install (TARGETS screen_dirt RUNTIME DESTINATION bin)
