#version 450

layout(push_constant) uniform VertPushConstants {
  mat4 model;
  mat4 view;
  mat4 proj;
} c;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
  gl_Position = c.proj * c.view * c.model * vec4(inPosition, 1.0);
  fragColor = inColor;
}
