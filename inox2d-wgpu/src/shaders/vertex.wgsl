struct CameraUniform { mvp: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> camera: CameraUniform;
struct TransformUniform { mvp: mat4x4<f32>, };
@group(3) @binding(0) var<uniform> transform: TransformUniform;
struct OriginUniform { origin: vec2<f32>, };
@group(4) @binding(0) var<uniform> origin: OriginUniform;
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex_albedo: texture_2d<f32>;
@group(1) @binding(2) var tex_emissive: texture_2d<f32>;
@group(1) @binding(3) var tex_bump: texture_2d<f32>;

struct VertexIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) deform: vec2<f32>,
};
struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragUniform {
    mult_color: vec4<f32>,
    screen_color: vec4<f32>,
    params: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = camera.mvp * transform.mvp * vec4<f32>(v.pos - origin.origin + v.deform, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}
