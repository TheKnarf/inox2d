use glam::{Mat4, UVec2};
use inox2d::math::camera::Camera;
use inox2d::model::Model;
use inox2d::node::{
    components::{Mask, Masks, TexturedMesh},
    drawables::{CompositeComponents, TexturedMeshComponents},
    InoxNodeUuid,
};
use inox2d::puppet::Puppet;
use inox2d::render::{CompositeRenderCtx, InoxRenderer, TexturedMeshRenderCtx};
use inox2d::texture::decode_model_textures;
use wgpu::util::DeviceExt;
use thiserror::Error;

const VERT_SHADER: &str = r#"
struct CameraUniform { mvp: mat4x4<f32>; };
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex: texture_2d<f32>;

struct VertexIn {
    @location(0) pos: vec2<f32>;
    @location(1) uv: vec2<f32>;
    @location(2) deform: vec2<f32>;
};
struct VertexOut {
    @builtin(position) pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = camera.mvp * vec4<f32>(v.pos + v.deform, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}
"#;

const FRAG_SHADER: &str = r#"
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

#[derive(Debug, Error)]
pub enum WgpuRendererError {
	#[error("wgpu error: {0}")]
	Wgpu(String),
}

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_len: u32,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    textures: Vec<wgpu::BindGroup>,
    dummy_view: wgpu::TextureView,
    pub camera: Camera,
    pub viewport: UVec2,
}

impl WgpuRenderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, model: &Model) -> Result<Self, WgpuRendererError> {
        let verts = model
            .puppet
            .render_ctx
            .as_ref()
            .expect("render ctx")
            .vertex_buffers
            .verts
            .clone();
        let uvs = &model
            .puppet
            .render_ctx
            .as_ref()
            .unwrap()
            .vertex_buffers
            .uvs;
        let deforms = &model
            .puppet
            .render_ctx
            .as_ref()
            .unwrap()
            .vertex_buffers
            .deforms;
        let mut vertices: Vec<[f32; 2 + 2 + 2]> = Vec::with_capacity(verts.len());
        for i in 0..verts.len() {
            vertices.push([
                verts[i].x,
                verts[i].y,
                uvs[i].x,
                uvs[i].y,
                deforms[i].x,
                deforms[i].y,
            ]);
        }
        let indices = &model
            .puppet
            .render_ctx
            .as_ref()
            .unwrap()
            .vertex_buffers
            .indices;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inox2d_verts"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inox2d_indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_len = indices.len() as u32;

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inox2d_camera"),
            size: std::mem::size_of::<Mat4>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inox2d_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inox2d_camera_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let textures = decode_model_textures(model.textures.iter())
            .iter()
            .map(|tex| {
                let size = wgpu::Extent3d {
                    width: tex.width(),
                    height: tex.height(),
                    depth_or_array_layers: 1,
                };
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("inox2d_texture"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    tex.pixels(),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width()),
                        rows_per_image: Some(tex.height()),
                    },
                    size,
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("inox2d_texture_bind_group"),
                    layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("inox2d_texture_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                        ],
                    }),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&sampler) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                    ],
                });
                bg
            })
            .collect::<Vec<_>>();

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("inox2d_shader"),
            source: wgpu::ShaderSource::Wgsl(VERT_SHADER.into()),
        });

        let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("inox2d_frag"),
            source: wgpu::ShaderSource::Wgsl(FRAG_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inox2d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout, &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inox2d_texture_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            })],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("inox2d_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("inox2d_dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            device,
            queue,
            vertex_buffer,
            index_buffer,
            index_len,
            camera_buf,
            camera_bg,
            bind_group_layout,
            pipeline,
            textures,
            dummy_view,
            camera: Camera::default(),
            viewport: UVec2::ZERO,
        })
    }

	pub fn resize(&mut self, width: u32, height: u32) {
		self.viewport = UVec2::new(width, height);
	}

        pub fn on_begin_draw(&self, _puppet: &Puppet) {
            let mvp = self.camera.matrix(self.viewport.as_vec2());
            let arr = mvp.to_cols_array();
            self.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&arr));
        }
	pub fn on_end_draw(&self, _puppet: &Puppet) {}
}

impl InoxRenderer for WgpuRenderer {
	fn on_begin_masks(&self, _masks: &Masks) {}
	fn on_begin_mask(&self, _mask: &Mask) {}
	fn on_begin_masked_content(&self) {}
	fn on_end_mask(&self) {}

        fn draw_textured_mesh_content(
                &self,
                _as_mask: bool,
                components: &TexturedMeshComponents,
                render_ctx: &TexturedMeshRenderCtx,
                _id: InoxNodeUuid,
        ) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inox2d_pass") });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("inox2d_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.dummy_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                let tex = components.texture.tex_albedo.raw();
                pass.set_bind_group(1, &self.textures[tex], &[]);
                let start = render_ctx.index_offset as u32;
                let end = start + render_ctx.index_len as u32;
                pass.draw_indexed(start..end, 0, 0..1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

	fn begin_composite_content(
		&self,
		_as_mask: bool,
		_components: &CompositeComponents,
		_render_ctx: &CompositeRenderCtx,
		_id: InoxNodeUuid,
	) {
	}

	fn finish_composite_content(
		&self,
		_as_mask: bool,
		_components: &CompositeComponents,
		_render_ctx: &CompositeRenderCtx,
		_id: InoxNodeUuid,
	) {
	}
}
