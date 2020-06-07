#version 450
#extension GL_ARB_separate_shader_objects : enable

// The vertex shader takes input from a vertex buffer using the in keyword.
layout(location = 0) in vec2 inPosition; // vertex position attribute
layout(location = 1) in vec3 inColor; // vertex color attribute

layout(location = 0) out vec3 fragColor;

void main() {
  gl_Position = vec4(inPosition, 0.0, 1.0);
  fragColor = inColor;
}