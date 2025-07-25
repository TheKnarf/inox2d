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
use std::collections::HashMap;
use thiserror::Error;
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
#[cfg(feature = "headless")]
pub mod headless;
mod shaders;

fn align_to(value: u32, alignment: u32) -> u32 {
	let remainder = value % alignment;
	if remainder == 0 {
		value
	} else {
		value + alignment - remainder
	}
}

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
	transform_buf: wgpu::Buffer,
	transform_bg: wgpu::BindGroup,
	origin_buf: wgpu::Buffer,
	origin_bg: wgpu::BindGroup,
	texture_layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
	frag_buf: wgpu::Buffer,
	frag_bg: wgpu::BindGroup,
	mask_buf: wgpu::Buffer,
	mask_bg: wgpu::BindGroup,
	pipelines: Vec<wgpu::RenderPipeline>,
	mask_pipeline: wgpu::RenderPipeline,
	stencil_ref: Cell<u32>,
	textures: Vec<wgpu::Texture>,
	texture_views: Vec<wgpu::TextureView>,
	texture_bind_groups: RefCell<HashMap<(usize, usize, usize), wgpu::BindGroup>>,
	composite_texture: RefCell<Option<wgpu::Texture>>,
	composite_view: RefCell<Option<wgpu::TextureView>>,
	composite_bg: RefCell<Option<wgpu::BindGroup>>,
	prev_target_view: RefCell<Option<wgpu::TextureView>>,
	stencil_texture: wgpu::Texture,
	stencil_view: wgpu::TextureView,
	offscreen_texture: wgpu::Texture,
	offscreen_view: wgpu::TextureView,
	using_offscreen: Cell<bool>,
	target_view: RefCell<Option<wgpu::TextureView>>,
	output_format: wgpu::TextureFormat,
	pub camera: Camera,
	pub viewport: UVec2,
	pub blend_mode: BlendMode,
	pub tint: Vec3,
	pub emission_strength: f32,
	pub mask_threshold: f32,
	debug_vertex_buffer: wgpu::Buffer,
	debug_index_buffer: wgpu::Buffer,
	debug_pipeline: wgpu::RenderPipeline,
	encoder: RefCell<Option<wgpu::CommandEncoder>>,
	render_pass: RefCell<Option<wgpu::RenderPass<'static>>>,
}

// WgpuRenderer interacts exclusively with the render thread. The underlying
// wgpu types are thread-safe, but interior pointers stored in `Cell` and
// `RefCell` mean the compiler can't automatically prove Send/Sync. Rendering
// occurs on a single thread, so declaring these as safe is acceptable here.
unsafe impl Send for WgpuRenderer {}
unsafe impl Sync for WgpuRenderer {}

impl WgpuRenderer {
	pub fn new(
		device: wgpu::Device,
		queue: wgpu::Queue,
		model: &Model,
		output_format: wgpu::TextureFormat,
	) -> Result<Self, WgpuRendererError> {
		tracing::info!("Initializing Inox2D renderer");
		tracing::debug!("Output format: {:?}", output_format);
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
		tracing::debug!(
			"Vertex buffer size: {} vertices, {} indices",
			verts.len(),
			indices.len()
		);

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

		let transform_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("inox2d_transform"),
			size: std::mem::size_of::<Mat4>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let transform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("inox2d_transform_layout"),
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
		let transform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("inox2d_transform_bg"),
			layout: &transform_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: transform_buf.as_entire_binding(),
			}],
		});

		let origin_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("inox2d_origin"),
			size: 16,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let origin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("inox2d_origin_layout"),
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
		let origin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("inox2d_origin_bg"),
			layout: &origin_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: origin_buf.as_entire_binding(),
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

		let decoded = decode_model_textures(model.textures.iter());
		let mut textures = Vec::with_capacity(decoded.len());
		let mut texture_views = Vec::with_capacity(decoded.len());
		for tex in decoded.iter() {
			let size = wgpu::Extent3d {
				width: tex.width(),
				height: tex.height(),
				depth_or_array_layers: 1,
			};
			// Textures returned from `decode_model_textures` are in
			// RGBA8 format and use sRGB color space. Use a matching
			// GPU format so we can upload bytes directly.
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

			// wgpu requires each row written to the GPU to be aligned
			// to COPY_BYTES_PER_ROW_ALIGNMENT (currently 256 bytes). Pad
			// the pixel data accordingly before uploading.
			let bytes_per_row = align_to(4 * tex.width(), wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
			debug_assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
			debug_assert!(bytes_per_row >= 4 * tex.width());
			let expected_len = 4 * tex.width() * tex.height();
			if tex.pixels().len() as u32 != expected_len {
				tracing::error!(
					"Texture pixel buffer length mismatch: expected {}, got {}",
					expected_len,
					tex.pixels().len()
				);
				continue;
			}

			let mut data = vec![0u8; (bytes_per_row * tex.height()) as usize];
			let row_bytes = (4 * tex.width()) as usize;
			// Each image row contains `row_bytes` bytes of RGBA8 data.
			let pixels = tex.pixels();
			for row in 0..tex.height() as usize {
				let src_start = row * row_bytes;
				let src_end = src_start + row_bytes;
				let dst_start = row * bytes_per_row as usize;
				data[dst_start..dst_start + row_bytes].copy_from_slice(&pixels[src_start..src_end]);
			}

			queue.write_texture(
				wgpu::TexelCopyTextureInfo {
					texture: &texture,
					mip_level: 0,
					origin: wgpu::Origin3d::ZERO,
					aspect: wgpu::TextureAspect::All,
				},
				&data,
				wgpu::TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(bytes_per_row),
					rows_per_image: Some(tex.height()),
				},
				size,
			);
			let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
			textures.push(texture);
			texture_views.push(view);
		}
		tracing::debug!("Loaded {} textures", texture_views.len());

		let texture_bind_groups = RefCell::new(HashMap::new());

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
			size: 32,
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
			source: wgpu::ShaderSource::Wgsl(shaders::vertex::SOURCE.into()),
		});

		let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_frag"),
			source: wgpu::ShaderSource::Wgsl(shaders::fragment::SOURCE.into()),
		});
		let mask_fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_mask_frag"),
			source: wgpu::ShaderSource::Wgsl(shaders::mask::SOURCE.into()),
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("inox2d_pipeline_layout"),
			bind_group_layouts: &[
				&bind_group_layout,
				&texture_layout,
				&frag_layout,
				&transform_layout,
				&origin_layout,
			],
			push_constant_ranges: &[],
		});
		let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("inox2d_mask_pipeline_layout"),
			bind_group_layouts: &[
				&bind_group_layout,
				&texture_layout,
				&mask_layout,
				&transform_layout,
				&origin_layout,
			],
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
						format: output_format,
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
		tracing::debug!("Created {} pipelines", pipelines.len());

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
					format: output_format,
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
						fail_op: wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
						pass_op: wgpu::StencilOperation::Replace,
					},
					back: wgpu::StencilFaceState {
						compare: wgpu::CompareFunction::Always,
						fail_op: wgpu::StencilOperation::Keep,
						depth_fail_op: wgpu::StencilOperation::Keep,
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
		tracing::debug!("Mask pipeline created");

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

		let dbg_vertices: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
		let dbg_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
		let debug_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("inox2d_debug_verts"),
			contents: bytemuck::cast_slice(&dbg_vertices),
			usage: wgpu::BufferUsages::VERTEX,
		});
		let debug_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("inox2d_debug_indices"),
			contents: bytemuck::cast_slice(&dbg_indices),
			usage: wgpu::BufferUsages::INDEX,
		});

		let debug_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("inox2d_debug_shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shaders/debug.wgsl").into()),
		});
		let debug_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("inox2d_debug_pipeline_layout"),
			bind_group_layouts: &[],
			push_constant_ranges: &[],
		});
		let debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("inox2d_debug_pipeline"),
			layout: Some(&debug_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &debug_shader,
				entry_point: Some("vs_main"),
				compilation_options: Default::default(),
				buffers: &[wgpu::VertexBufferLayout {
					array_stride: 8,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: &wgpu::vertex_attr_array![0 => Float32x2],
				}],
			},
			fragment: Some(wgpu::FragmentState {
				module: &debug_shader,
				entry_point: Some("fs_main"),
				compilation_options: Default::default(),
				targets: &[Some(wgpu::ColorTargetState {
					format: output_format,
					blend: Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState::default(),
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			cache: None,
		});

		tracing::info!("Inox2D renderer initialized");
		let offscreen_texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("inox2d_offscreen"),
			size: wgpu::Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: output_format,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT
				| wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		let offscreen_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());

		let renderer = Self {
			device,
			queue,
			vertex_buffer,
			index_buffer,
			camera_buf,
			camera_bg,
			transform_buf,
			transform_bg,
			origin_buf,
			origin_bg,
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
			texture_views,
			texture_bind_groups,
			composite_texture: RefCell::new(None),
			composite_view: RefCell::new(None),
			composite_bg: RefCell::new(None),
			prev_target_view: RefCell::new(None),
			stencil_texture,
			stencil_view,
			offscreen_texture,
			offscreen_view,
			using_offscreen: Cell::new(true),
			target_view: RefCell::new(None),
			output_format,
			camera: Camera::default(),
			viewport: UVec2::ZERO,
			blend_mode: BlendMode::Normal,
			tint: Vec3::ONE,
			emission_strength: 1.0,
			mask_threshold: 0.5,
			debug_vertex_buffer,
			debug_index_buffer,
			debug_pipeline,
			encoder: RefCell::new(None),
			render_pass: RefCell::new(None),
		};
		*renderer.target_view.borrow_mut() = Some(renderer.offscreen_view.clone());
		Ok(renderer)
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

		self.offscreen_texture = self.device.create_texture(&wgpu::TextureDescriptor {
			label: Some("inox2d_offscreen"),
			size: wgpu::Extent3d {
				width: width.max(1),
				height: height.max(1),
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: self.output_format,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT
				| wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		self.offscreen_view = self
			.offscreen_texture
			.create_view(&wgpu::TextureViewDescriptor::default());

		if self.using_offscreen.get() || self.target_view.borrow().is_none() {
			*self.target_view.borrow_mut() = Some(self.offscreen_view.clone());
			self.using_offscreen.set(true);
		}
	}

	pub fn set_target_view(&self, view: &wgpu::TextureView) {
		*self.target_view.borrow_mut() = Some(view.clone());
		self.using_offscreen.set(false);
	}

	pub fn target_view(&self) -> std::cell::Ref<'_, wgpu::TextureView> {
		std::cell::Ref::map(self.target_view.borrow(), |v| v.as_ref().expect("target view not set"))
	}

	pub fn copy_target_to_buffer(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		texture: &wgpu::Texture,
		buffer: &wgpu::Buffer,
	) {
		let bytes_per_row = align_to(4 * self.viewport.x, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
		encoder.copy_texture_to_buffer(
			wgpu::ImageCopyTexture {
				texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyBuffer {
				buffer,
				layout: wgpu::ImageDataLayout {
					offset: 0,
					bytes_per_row: Some(bytes_per_row),
					rows_per_image: None,
				},
			},
			wgpu::Extent3d {
				width: self.viewport.x,
				height: self.viewport.y,
				depth_or_array_layers: 1,
			},
		);
	}

	pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
		let tv = self.target_view();
		encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("inox2d_clear"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: &*tv,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
					store: wgpu::StoreOp::Store,
				},
			})],
			depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
				view: &self.stencil_view,
				depth_ops: None,
				stencil_ops: Some(wgpu::Operations {
					load: wgpu::LoadOp::Clear(1),
					store: wgpu::StoreOp::Store,
				}),
			}),
			timestamp_writes: None,
			occlusion_query_set: None,
		});
	}

	pub fn on_begin_draw(&self, puppet: &Puppet) {
		tracing::debug!("Begin draw");
		let mut enc_ref = self.encoder.borrow_mut();
		let encoder = enc_ref.insert(self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_frame"),
		}));

		let tv = self.target_view();
		let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("inox2d_frame"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: &*tv,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
					store: wgpu::StoreOp::Store,
				},
			})],
			depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
				view: &self.stencil_view,
				depth_ops: None,
				stencil_ops: Some(wgpu::Operations {
					load: wgpu::LoadOp::Clear(1),
					store: wgpu::StoreOp::Store,
				}),
			}),
			timestamp_writes: None,
			occlusion_query_set: None,
		});
		*self.render_pass.borrow_mut() =
			Some(unsafe { std::mem::transmute::<wgpu::RenderPass<'_>, wgpu::RenderPass<'static>>(pass) });

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
	pub fn on_end_draw(&self, _puppet: &Puppet) {
		tracing::debug!("End draw");
		self.render_pass.borrow_mut().take();
		if let Some(encoder) = self.encoder.borrow_mut().take() {
			self.queue.submit(Some(encoder.finish()));
		}
	}

	pub fn draw_debug_rect(&self) {
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("inox2d_debug_rect"),
		});
		{
			let tv = self.target_view();
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_debug_rect"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &*tv,
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
			pass.set_pipeline(&self.debug_pipeline);
			pass.set_vertex_buffer(0, self.debug_vertex_buffer.slice(..));
			pass.set_index_buffer(self.debug_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			pass.draw_indexed(0..6, 0, 0..1);
		}
		self.queue.submit(Some(encoder.finish()));
	}
}

impl InoxRenderer for WgpuRenderer {
	fn on_begin_masks(&self, masks: &Masks) {
		let clear_val = if masks.has_masks() { 0 } else { 1 };
		let threshold = (masks.threshold * self.mask_threshold).clamp(0.0, 1.0);
		self.queue
			.write_buffer(&self.mask_buf, 0, bytemuck::bytes_of(&threshold));

		self.render_pass.borrow_mut().take();
		let mut enc_ref = self.encoder.borrow_mut();
		let encoder = enc_ref.as_mut().expect("on_begin_draw not called");

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

		let tv = self.target_view();

		let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("inox2d_pass"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: &*tv,
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
		*self.render_pass.borrow_mut() =
			Some(unsafe { std::mem::transmute::<wgpu::RenderPass<'_>, wgpu::RenderPass<'static>>(pass) });
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
		let arr = components.transform.to_cols_array();

		tracing::debug!(
			"vertex buffer size: {}, index buffer size: {}",
			self.vertex_buffer.size(),
			self.index_buffer.size()
		);

		self.queue
			.write_buffer(&self.transform_buf, 0, bytemuck::cast_slice(&arr));
		let zero = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
		self.queue
			.write_buffer(&self.origin_buf, 0, bytemuck::cast_slice(&zero));
		let origin = components.mesh.origin;
		let origin_arr = [origin.x, origin.y, 0.0, 0.0];
		self.queue
			.write_buffer(&self.origin_buf, 0, bytemuck::cast_slice(&origin_arr));

		let mut pass_ref = self.render_pass.borrow_mut();
		let pass = pass_ref.as_mut().expect("on_begin_draw not called");
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
		pass.set_bind_group(3, &self.transform_bg, &[]);
		pass.set_bind_group(4, &self.origin_bg, &[]);
		let albedo = components.texture.tex_albedo.raw();
		let emissive = components.texture.tex_emissive.raw();
		let bump = components.texture.tex_bumpmap.raw();
		let key = (albedo, emissive, bump);
		let mut tex_bgs = self.texture_bind_groups.borrow_mut();
		let tex_bg = tex_bgs.entry(key).or_insert_with(|| {
			self.device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("inox2d_texture_set"),
				layout: &self.texture_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::Sampler(&self.sampler),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::TextureView(&self.texture_views[albedo]),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: wgpu::BindingResource::TextureView(&self.texture_views[emissive]),
					},
					wgpu::BindGroupEntry {
						binding: 3,
						resource: wgpu::BindingResource::TextureView(&self.texture_views[bump]),
					},
				],
			})
		});
		pass.set_bind_group(1, &*tex_bg, &[]);
		let start = render_ctx.index_offset as u32;
		let end = start + render_ctx.index_len as u32;
		tracing::debug!(
			"draw_indexed range: {start:?}..{end:?}, stencil_ref: {}",
			self.stencil_ref.get()
		);
		pass.draw_indexed(start..end, 0, 0..1);
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
			format: self.output_format,
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
		*self.composite_view.borrow_mut() = Some(view.clone());
		*self.composite_bg.borrow_mut() = Some(bg);
		*self.prev_target_view.borrow_mut() = self.target_view.borrow().clone();
		*self.target_view.borrow_mut() = Some(view);
		self.render_pass.borrow_mut().take();
		let mut enc_ref = self.encoder.borrow_mut();
		let encoder = enc_ref.as_mut().expect("on_begin_draw not called");
		let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("inox2d_clear_composite"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: self.composite_view.borrow().as_ref().unwrap(),
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
		*self.render_pass.borrow_mut() =
			Some(unsafe { std::mem::transmute::<wgpu::RenderPass<'_>, wgpu::RenderPass<'static>>(pass) });
	}

	fn finish_composite_content(
		&self,
		_as_mask: bool,
		components: &CompositeComponents,
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
		if let Some(prev_view) = self.prev_target_view.borrow_mut().take() {
			*self.target_view.borrow_mut() = Some(prev_view.clone());

			let arr = components.transform.to_cols_array();
			self.queue
				.write_buffer(&self.transform_buf, 0, bytemuck::cast_slice(&arr));
			let zero = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
			self.queue
				.write_buffer(&self.origin_buf, 0, bytemuck::cast_slice(&zero));

			self.render_pass.borrow_mut().take();
			let mut enc_ref = self.encoder.borrow_mut();
			let encoder = enc_ref.as_mut().expect("on_begin_draw not called");
			let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("inox2d_composite_blend"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &prev_view,
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

			let mut pass_ref = self.render_pass.borrow_mut();
			let pass = pass_ref
				.insert(unsafe { std::mem::transmute::<wgpu::RenderPass<'_>, wgpu::RenderPass<'static>>(pass) });
			pass.set_pipeline(&self.pipelines[BlendMode::Normal as usize]);
			pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			pass.set_bind_group(0, &self.camera_bg, &[]);
			pass.set_bind_group(1, &bg, &[]);
			pass.set_bind_group(2, &self.frag_bg, &[]);
			pass.set_bind_group(3, &self.transform_bg, &[]);
			pass.set_bind_group(4, &self.origin_bg, &[]);
			pass.draw_indexed(0..6, 0, 0..1);

			*self.composite_texture.borrow_mut() = None;
			*self.composite_view.borrow_mut() = None;
			*self.composite_bg.borrow_mut() = None;
		}
	}
}
