struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex_albedo: texture_2d<f32>;

struct MaskUniform {
    threshold: f32,
    // Padding to satisfy 16-byte alignment requirements
    _pad: vec3<f32>,
};
@group(2) @binding(0) var<uniform> mask: MaskUniform;

@fragment
fn fs_mask(in: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSample(tex_albedo, samp, in.uv);
    if (color.a <= mask.threshold) {
        discard;
    }
    return vec4<f32>(1.0);
}
