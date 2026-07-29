#version 450

layout(location = 0) out vec3 vertexColor;

const vec2 positions[3] = vec2[](
    vec2(-0.8, -0.8),
    vec2( 0.8, -0.8),
    vec2( 0.0,  0.8)
);

const vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    vertexColor = colors[gl_VertexIndex];
}
