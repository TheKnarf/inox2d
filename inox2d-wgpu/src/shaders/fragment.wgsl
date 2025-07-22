struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragUniform {
    mult_color: vec4<f32>,
    screen_color: vec4<f32>,
    params: vec4<f32>,
};
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex_albedo: texture_2d<f32>;
@group(1) @binding(2) var tex_emissive: texture_2d<f32>;
@group(1) @binding(3) var tex_bump: texture_2d<f32>;
@group(2) @binding(0) var<uniform> frag: FragUniform;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex_albedo, samp, in.uv);
    let screen_out = vec3<f32>(1.0) - ((vec3<f32>(1.0) - tex_color.xyz) * (vec3<f32>(1.0) - (frag.screen_color.xyz * tex_color.a)));
    var base = vec4<f32>(screen_out, tex_color.a) * frag.mult_color * frag.params.x;
    let emissive = textureSample(tex_emissive, samp, in.uv).xyz * frag.params.y * base.a;
    let _bump = textureSample(tex_bump, samp, in.uv); // currently unused
    base = vec4<f32>(base.xyz + emissive, base.a);
    return base;
}
