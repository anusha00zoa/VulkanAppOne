#version 450
#extension GL_ARB_separate_shader_objects : enable

// The vertex shader takes input from a vertex buffer using the in keyword.
layout(location = 0) in vec3 inPosition; // vertex position attribute
layout(location = 1) in vec3 inColor; // vertex color attribute

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
  // each mat4 is 4x4x4 = 64 bytes
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

void main() {
  gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);
  fragColor = inColor;
}

/// NOTES on Alignment Requirements
/// Vulkan expects the data in your structure to be aligned in memory in a specific way:
/// 1. Scalars have to be aligned by N (= 4 bytes given 32 bit floats).
/// 2. A vec2 must be aligned by 2N (= 8 bytes)
/// 3. A vec3 or vec4 must be aligned by 4N (= 16 bytes)
/// 4. A nested structure must be aligned by the base alignment of its members rounded up to a
/// multiple of 16.
/// 5. A mat4 matrix must have the same alignment as a vec4.
/// To fix alignment issues, use 'alignas'