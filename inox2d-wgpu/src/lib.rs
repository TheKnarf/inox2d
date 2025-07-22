use glam::{Mat4, UVec2, Vec3};
use inox2d::math::camera::Camera;
use inox2d::model::Model;
use inox2d::node::{
	components::{BlendMode, Mask, MaskMode, Masks},
	drawables::{CompositeComponents, TexturedMeshComponents},
	InoxNodeUuid,
};
use inox2d::puppet::Puppet;
use inox2d::render::{CompositeRenderCtx, InoxRenderer, TexturedMeshRenderCtx};
use inox2d::texture::decode_model_textures;
use std::cell::{Cell, RefCell};
use thiserror::Error;
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
const VERT_SHADER: &str = r#"
struct CameraUniform { mvp: mat4x4<f32>; };
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex_albedo: texture_2d<f32>;
@group(1) @binding(2) var tex_emissive: texture_2d<f32>;
@group(1) @binding(3) var tex_bump: texture_2d<f32>;

struct VertexIn {
    @location(0) pos: vec2<f32>;
    @location(1) uv: vec2<f32>;
    @location(2) deform: vec2<f32>;
};
struct VertexOut {
    @builtin(position) pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

struct FragUniform {
    mult_color: vec4<f32>;
    screen_color: vec4<f32>;
    params: vec4<f32>;
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
"#;

const MASK_FRAG_SHADER: &str = r#"
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex_albedo: texture_2d<f32>;

struct MaskUniform { threshold: f32; };
@group(2) @binding(0) var<uniform> mask: MaskUniform;

@fragment
fn fs_mask(in: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSample(tex_albedo, samp, in.uv);
    if (color.a <= mask.threshold) {
        discard;
    }
    return vec4<f32>(1.0);
}
"#;

fn blend_state_for(mode: BlendMode) -> wgpu::BlendState {
	use wgpu::{BlendComponent, BlendFactor as F, BlendOperation as O};
	let comp = |src, dst, op| BlendComponent {
		src_factor: src,
		dst_factor: dst,
		operation: op,
	};
	match mode {
		BlendMode::Normal => wgpu::BlendState::ALPHA_BLENDING,
		BlendMode::Multiply => wgpu::BlendState {
			color: comp(F::Dst, F::OneMinusSrcAlpha, O::Add),
			alpha: comp(F::Dst, F::OneMinusSrcAlpha, O::Add),
		},
		BlendMode::ColorDodge => wgpu::BlendState {
			color: comp(F::Dst, F::One, O::Add),
			alpha: comp(F::Dst, F::One, O::Add),
		},
		BlendMode::LinearDodge => wgpu::BlendState {
			color: comp(F::One, F::One, O::Add),
			alpha: comp(F::One, F::One, O::Add),
		},
		BlendMode::Screen => wgpu::BlendState {
			color: comp(F::One, F::OneMinusSrc, O::Add),
			alpha: comp(F::One, F::OneMinusSrc, O::Add),
		},
		BlendMode::ClipToLower => wgpu::BlendState {
			color: comp(F::DstAlpha, F::OneMinusSrcAlpha, O::Add),
			alpha: comp(F::DstAlpha, F::OneMinusSrcAlpha, O::Add),
		},
		BlendMode::SliceFromLower => wgpu::BlendState {
			color: comp(F::OneMinusDstAlpha, F::OneMinusSrcAlpha, O::Subtract),
			alpha: comp(F::OneMinusDstAlpha, F::OneMinusSrcAlpha, O::Subtract),
		},
	}
}

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
	camera_buf: wgpu::Buffer,
	camera_bg: wgpu::BindGroup,
	texture_layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
	frag_buf: wgpu::Buffer,
	frag_bg: wgpu::BindGroup,
	mask_buf: wgpu::Buffer,
	mask_bg: wgpu::BindGroup,
	pipelines: Vec<wgpu::RenderPipeline>,
	mask_pipeline: wgpu::RenderPipeline,
	stencil_ref: Cell<u32>,
	textures: Vec<wgpu::TextureView>,
	composite_texture: RefCell<Option<wgpu::Texture>>,
	composite_view: RefCell<Option<wgpu::TextureView>>,
	composite_bg: RefCell<Option<wgpu::BindGroup>>,
	prev_target_view: Cell<*const wgpu::TextureView>,
	stencil_texture: wgpu::Texture,
	stencil_view: wgpu::TextureView,
	target_view: Cell<*const wgpu::TextureView>,
	pub camera: Camera,
	pub viewport: UVec2,
	pub blend_mode: BlendMode,
	pub tint: Vec3,
	pub emission_strength: f32,
	pub mask_threshold: f32,
}

// WgpuRenderer interacts exclusively with the render thread. The underlying
// wgpu types are thread-safe, but interior pointers stored in `Cell` and
// `RefCell` mean the compiler can't automatically prove Send/Sync. Rendering
// occurs on a single thread, so declaring these as safe is acceptable here.
unsafe impl Send for WgpuRenderer {}
unsafe impl Sync for WgpuRenderer {}

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
		let uvs = &model.puppet.render_ctx.as_ref().unwrap().vertex_buffers.uvs;
		let deforms = &model.puppet.render_ctx.as_ref().unwrap().vertex_buffers.deforms;
		let mut vertices: Vec<[f32; 2 + 2 + 2]> = Vec::with_capacity(verts.len());
		for i in 0..verts.len() {
			vertices.push([verts[i].x, verts[i].y, uvs[i].x, uvs[i].y, deforms[i].x, deforms[i].y]);
		}
		let indices = &model.puppet.render_ctx.as_ref().unwrap().vertex_buffers.indices;

		let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("inox2d_verts"),
			contents: bytemuck::cast_slice(&vertices),
			usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
		});
		let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("inox2d_indices"),
			contents: bytemuck::cast_slice(indices),
			usage: wgpu::BufferUsages::INDEX,
		});

		let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("inox2d_camera"),
			size: std::mem::size_of::<Mat4>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("inox2d_bind_group_layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
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
		let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
				wgpu::BindGroupLayoutEntry {
					binding: 2,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 3,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
			],
		});

		let frag_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("inox2d_frag_layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::FRAGMENT,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});

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
					wgpu::TexelCopyTextureInfo {
						texture: &texture,
						mip_level: 0,
						origin: wgpu::Origin3d::ZERO,
						aspect: wgpu::TextureAspect::All,
					},
					tex.pixels(),
					wgpu::TexelCopyBufferLayout {
						offset: 0,
						bytes_per_row: Some(4 * tex.width()),
						rows_per_image: Some(tex.height()),
					},
					size,
				);
				let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
				// keep sampler separate, bind groups created per draw
				view
			})
			.collect::<Vec<_>>();

		let frag_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("inox2d_frag_buf"),
			size: 48,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let frag_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("inox2d_frag_bg"),
			layout: &frag_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: frag_buf.as_entire_binding(),
			}],
		});

		let mask_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("inox2d_mask_layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::FRAGMENT,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});
		let mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("inox2d_mask_buf"),
			size: 4,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let mask_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("inox2d_mask_bg"),
			layout: &mask_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: mask_buf.as_entire_binding(),
			}],
		});

		let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_shader"),
			source: wgpu::ShaderSource::Wgsl(VERT_SHADER.into()),
		});

		let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_frag"),
			source: wgpu::ShaderSource::Wgsl(FRAG_SHADER.into()),
		});
		let mask_fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_mask_frag"),
			source: wgpu::ShaderSource::Wgsl(MASK_FRAG_SHADER.into()),
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("inox2d_pipeline_layout"),
			bind_group_layouts: &[&bind_group_layout, &texture_layout, &frag_layout],
			push_constant_ranges: &[],
		});
		let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("inox2d_mask_pipeline_layout"),
			bind_group_layouts: &[&bind_group_layout, &texture_layout, &mask_layout],
			push_constant_ranges: &[],
		});

		let mut pipelines = Vec::new();
		for mode in BlendMode::VALUES {
			let blend = blend_state_for(mode);
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
						blend: Some(blend),
						write_mask: wgpu::ColorWrites::ALL,
					})],
				}),
				primitive: wgpu::PrimitiveState::default(),
				depth_stencil: Some(wgpu::DepthStencilState {
					format: DEPTH_FORMAT,
					depth_write_enabled: false,
					depth_compare: wgpu::CompareFunction::Always,
					stencil: wgpu::StencilState {
						front: wgpu::StencilFaceState {
							compare: wgpu::CompareFunction::Equal,
							fail_op: wgpu::StencilOperation::Keep,
							depth_fail_op: wgpu::StencilOperation::Keep,
							pass_op: wgpu::StencilOperation::Keep,
						},
						back: wgpu::StencilFaceState {
							compare: wgpu::CompareFunction::Equal,
							fail_op: wgpu::StencilOperation::Keep,
							depth_fail_op: wgpu::StencilOperation::Keep,
							pass_op: wgpu::StencilOperation::Keep,
						},
						read_mask: 0xff,
						write_mask: 0x00,
					},
					bias: wgpu::DepthBiasState::default(),
				}),
				multisample: wgpu::MultisampleState::default(),
				multiview: None,
				cache: None,
			});
			pipelines.push(pipeline);
		}

		let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("inox2d_mask_pipeline"),
			layout: Some(&mask_pipeline_layout),
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
				module: &mask_fragment_module,
				entry_point: Some("fs_mask"),
				compilation_options: Default::default(),
				targets: &[Some(wgpu::ColorTargetState {
					format: wgpu::TextureFormat::Bgra8UnormSrgb,
					blend: Some(wgpu::BlendState::ALPHA_BLENDING),
					write_mask: wgpu::ColorWrites::empty(),
				})],
			}),
			primitive: wgpu::PrimitiveState::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format: DEPTH_FORMAT,
				depth_write_enabled: false,
				depth_compare: wgpu::CompareFunction::Always,
				stencil: wgpu::StencilState {
					front: wgpu::StencilFaceState {
						compare: wgpu::CompareFunction::Always,
						fail_op: wgpu::StencilOperation::Replace,
						depth_fail_op: wgpu::StencilOperation::Replace,
						pass_op: wgpu::StencilOperation::Replace,
					},
					back: wgpu::StencilFaceState {
						compare: wgpu::CompareFunction::Always,
						fail_op: wgpu::StencilOperation::Replace,
						depth_fail_op: wgpu::StencilOperation::Replace,
						pass_op: wgpu::StencilOperation::Replace,
					},
					read_mask: 0xff,
					write_mask: 0xff,
				},
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			cache: None,
		});

		let stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("inox2d_stencil"),
			size: wgpu::Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			view_formats: &[],
		});
		let stencil_view = stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

		Ok(Self {
			device,
			queue,
			vertex_buffer,
			index_buffer,
			camera_buf,
			camera_bg,
			texture_layout,
			sampler,
			frag_buf,
			frag_bg,
			mask_buf,
			mask_bg,
			pipelines,
			mask_pipeline,
			stencil_ref: Cell::new(1),
			textures,
			composite_texture: RefCell::new(None),
			composite_view: RefCell::new(None),
			composite_bg: RefCell::new(None),
			prev_target_view: Cell::new(core::ptr::null()),
			stencil_texture,
			stencil_view,
			target_view: Cell::new(core::ptr::null()),
			camera: Camera::default(),
			viewport: UVec2::ZERO,
			blend_mode: BlendMode::Normal,
			tint: Vec3::ONE,
			emission_strength: 1.0,
			mask_threshold: 0.5,
		})
	}

	pub fn resize(&mut self, width: u32, height: u32) {
		self.viewport = UVec2::new(width, height);
		self.stencil_texture = self.device.create_texture(&wgpu::TextureDescriptor {
			label: Some("inox2d_stencil"),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			view_formats: &[],
		});
		self.stencil_view = self
			.stencil_texture
			.create_view(&wgpu::TextureViewDescriptor::default());
	}

	pub fn set_target_view(&self, view: &wgpu::TextureView) {
		self.target_view.set(view as *const _);
	}

	pub fn on_begin_draw(&self, puppet: &Puppet) {
		let mvp = self.camera.matrix(self.viewport.as_vec2());
		let arr = mvp.to_cols_array();
		self.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&arr));

		let render_ctx = puppet
			.render_ctx
			.as_ref()
			.expect("Rendering for a puppet must be initialized by now.");
		let verts = &render_ctx.vertex_buffers.verts;
		let uvs = &render_ctx.vertex_buffers.uvs;
		let deforms = &render_ctx.vertex_buffers.deforms;
		let mut vertices: Vec<[f32; 6]> = Vec::with_capacity(verts.len());
		for i in 0..verts.len() {
			vertices.push([verts[i].x, verts[i].y, uvs[i].x, uvs[i].y, deforms[i].x, deforms[i].y]);
		}
		self.queue
			.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
	}
	pub fn on_end_draw(&self, _puppet: &Puppet) {}
}

impl InoxRenderer for WgpuRenderer {
	fn on_begin_masks(&self, masks: &Masks) {
		let clear_val = if masks.has_masks() { 0 } else { 1 };
		let threshold = (masks.threshold * self.mask_threshold).clamp(0.0, 1.0);
		self.queue
			.write_buffer(&self.mask_buf, 0, bytemuck::bytes_of(&threshold));
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_clear_stencil"),
		});
		{
			encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_clear_stencil"),
				color_attachments: &[],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					view: &self.stencil_view,
					depth_ops: None,
					stencil_ops: Some(wgpu::Operations {
						load: wgpu::LoadOp::Clear(clear_val),
						store: wgpu::StoreOp::Store,
					}),
				}),
				timestamp_writes: None,
				occlusion_query_set: None,
			});
		}
		self.queue.submit(Some(encoder.finish()));
	}
	fn on_begin_mask(&self, mask: &Mask) {
		let val = if mask.mode == MaskMode::Mask { 1 } else { 0 };
		self.stencil_ref.set(val);
	}
	fn on_begin_masked_content(&self) {
		self.stencil_ref.set(1);
	}
	fn on_end_mask(&self) {
		self.stencil_ref.set(1);
	}

	fn draw_textured_mesh_content(
		&self,
		as_mask: bool,
		components: &TexturedMeshComponents,
		render_ctx: &TexturedMeshRenderCtx,
		_id: InoxNodeUuid,
	) {
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_pass"),
		});
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_pass"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: unsafe { &*self.target_view.get() },
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					view: &self.stencil_view,
					depth_ops: None,
					stencil_ops: Some(wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: wgpu::StoreOp::Store,
					}),
				}),
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			if as_mask {
				pass.set_pipeline(&self.mask_pipeline);
				pass.set_bind_group(2, &self.mask_bg, &[]);
			} else {
				let idx = self.blend_mode as usize;
				pass.set_pipeline(&self.pipelines[idx]);
				let blend = &components.drawable.blending;
				let tint = blend.tint * self.tint;
				let data = [
					tint.x,
					tint.y,
					tint.z,
					1.0,
					blend.screen_tint.x,
					blend.screen_tint.y,
					blend.screen_tint.z,
					1.0,
					blend.opacity,
					self.emission_strength,
					0.0,
					0.0,
				];
				self.queue.write_buffer(&self.frag_buf, 0, bytemuck::cast_slice(&data));
				pass.set_bind_group(2, &self.frag_bg, &[]);
			}
			pass.set_stencil_reference(self.stencil_ref.get());
			pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			pass.set_bind_group(0, &self.camera_bg, &[]);
			let albedo = components.texture.tex_albedo.raw();
			let emissive = components.texture.tex_emissive.raw();
			let bump = components.texture.tex_bumpmap.raw();
			let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("inox2d_texture_set"),
				layout: &self.texture_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::Sampler(&self.sampler),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::TextureView(&self.textures[albedo]),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: wgpu::BindingResource::TextureView(&self.textures[emissive]),
					},
					wgpu::BindGroupEntry {
						binding: 3,
						resource: wgpu::BindingResource::TextureView(&self.textures[bump]),
					},
				],
			});
			pass.set_bind_group(1, &tex_bg, &[]);
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
		let size = wgpu::Extent3d {
			width: self.viewport.x.max(1),
			height: self.viewport.y.max(1),
			depth_or_array_layers: 1,
		};
		let texture = self.device.create_texture(&wgpu::TextureDescriptor {
			label: Some("inox2d_composite"),
			size,
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Bgra8UnormSrgb,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});
		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("inox2d_composite_bg"),
			layout: &self.texture_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&self.sampler),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&view),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::TextureView(&view),
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: wgpu::BindingResource::TextureView(&view),
				},
			],
		});
		*self.composite_texture.borrow_mut() = Some(texture);
		*self.composite_view.borrow_mut() = Some(view);
		*self.composite_bg.borrow_mut() = Some(bg);
		self.prev_target_view.set(self.target_view.get());
		self.target_view
			.set(self.composite_view.borrow().as_ref().unwrap() as *const _);

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_clear_composite"),
		});
		{
			encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_clear_composite"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: self.composite_view.borrow().as_ref().unwrap(),
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});
		}
		self.queue.submit(Some(encoder.finish()));
	}

	fn finish_composite_content(
		&self,
		_as_mask: bool,
		_components: &CompositeComponents,
		_render_ctx: &CompositeRenderCtx,
		_id: InoxNodeUuid,
	) {
		if self.composite_view.borrow().is_none() {
			return;
		}
		let bg = match self.composite_bg.borrow().clone() {
			Some(b) => b,
			None => return,
		};
		let prev = self.prev_target_view.get();
		self.target_view.set(prev);

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_composite_blend"),
		});
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_composite_blend"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: unsafe { &*prev },
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			pass.set_pipeline(&self.pipelines[BlendMode::Normal as usize]);
			pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			pass.set_bind_group(0, &self.camera_bg, &[]);
			pass.set_bind_group(1, &bg, &[]);
			pass.draw_indexed(0..6, 0, 0..1);
		}
		self.queue.submit(Some(encoder.finish()));

		*self.composite_texture.borrow_mut() = None;
		*self.composite_view.borrow_mut() = None;
		*self.composite_bg.borrow_mut() = None;
	}
}
