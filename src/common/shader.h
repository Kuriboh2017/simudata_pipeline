#ifndef SCREEN_EFFECTS_DIRT_SRC_SHADER_H_
#define SCREEN_EFFECTS_DIRT_SRC_SHADER_H_

#include <string>

// The class compiles and links the shader during the runtime.
class Shader
{
public:

    Shader(const char* vertexPath, const char* fragmentPath);
    void Use();

    unsigned int id() const {return id_;}
private:
    void CheckCompileErrors(unsigned int shader, std::string type);
    unsigned int id_;
};
#endif

