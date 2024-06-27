struct VertexInput {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

struct PushConstants {
  model: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
};
var<push_constant> c: PushConstants;

@vertex
fn vs_main(in: VertexInput)
    -> VertexOutput {
  var out: VertexOutput;
  out.color = in.color;
  var pos: vec4<f32> = vec4<f32>(in.pos, 1.0);
  out.clip_position = pos * c.model * c.view * c.proj;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  // return vec4<f32>(in.vert_pos.x, in.vert_pos.y, 0.0, 1.0);
  return vec4<f32>(in.color, 1.0);
}
