use std::path::PathBuf;
use std::{error::Error, fs};

use clap::Parser;
use glam::Vec2;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use wgpu::{Surface, SurfaceConfiguration};
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};

use common::scene::ExampleSceneController;
use inox2d::formats::inp::parse_inp;
use inox2d::render::InoxRendererExt;
use inox2d_wgpu::WgpuRenderer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
	#[arg(help = "Path to the .inp or .inx file.")]
	inp_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
	pollster::block_on(run())
}

async fn init_wgpu(
	window: &Window,
) -> Result<(Surface, wgpu::Device, wgpu::Queue, SurfaceConfiguration), Box<dyn Error>> {
	let size = window.inner_size();
	tracing::debug!("Initializing WGPU with window size: {:?}", size);

	let instance = wgpu::Instance::default();
	let surface = instance.create_surface(window)?;

	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::HighPerformance,
			compatible_surface: Some(&surface),
			force_fallback_adapter: false,
		})
		.await
		.ok_or("Failed to find an adapter")?;

	let info = adapter.get_info();
	tracing::info!("Using adapter: {} ({:?})", info.name, info.backend);
	tracing::debug!("Adapter features: {:?}", adapter.features());

	let limits = wgpu::Limits {
		max_bind_groups: 5,
		..wgpu::Limits::default()
	};
	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				required_features: wgpu::Features::empty(),
				required_limits: limits,
				..Default::default()
			},
			None,
		)
		.await?;

	tracing::debug!("Device limits: {:?}", device.limits());
	let caps = surface.get_capabilities(&adapter);
	let format = caps
		.formats
		.iter()
		.copied()
		.find(|f| f.is_srgb())
		.unwrap_or(caps.formats[0]);
	let config = SurfaceConfiguration {
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
		format,
		width: size.width.max(1),
		height: size.height.max(1),
		present_mode: wgpu::PresentMode::Fifo,
		desired_maximum_frame_latency: 2,
		alpha_mode: caps.alpha_modes[0],
		view_formats: vec![],
	};
	tracing::debug!("Surface format chosen: {:?}", format);
	surface.configure(&device, &config);
	tracing::info!(
		"Surface configured: {}x{} present_mode={:?}",
		config.width,
		config.height,
		config.present_mode
	);
	Ok((surface, device, queue, config))
}

async fn run() -> Result<(), Box<dyn Error>> {
	let cli = Cli::parse();

	let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
	tracing_subscriber::registry()
		.with(fmt::layer())
		.with(env_filter)
		.init();

	tracing::info!("Parsing puppet");
	let data = fs::read(cli.inp_path)?;
	let mut model = parse_inp(data.as_slice())?;
	tracing::info!(
		"Successfully parsed puppet: {}",
		(model.puppet.meta.name.as_deref()).unwrap_or("<no puppet name specified in file>")
	);

	tracing::info!("Setting up puppet for transforms, params and rendering.");
	model.puppet.init_transforms();
	model.puppet.init_rendering();
	model.puppet.init_params();
	model.puppet.init_physics();

	tracing::info!("Setting up windowing and WGPU");
	let event_loop = EventLoop::new()?;
	let window = WindowBuilder::new()
		.with_transparent(true)
		.with_resizable(true)
		.with_inner_size(winit::dpi::PhysicalSize::new(600, 800))
		.with_title("Render Inochi2D Puppet (WGPU)")
		.build(&event_loop)?;

	// Leak the window so the surface can outlive the original binding.
	let window: &'static Window = Box::leak(Box::new(window));

	let (surface, device, queue, mut surface_config) = init_wgpu(window).await?;
	// Store the registration so the callback lives for the entire program
	let _error_callback = device.on_uncaptured_error(Box::new(|e| {
		tracing::error!("wgpu uncaptured error: {:?}", e);
	}));

	tracing::info!("Initializing Inox2D renderer");
	let mut renderer = WgpuRenderer::new(device.clone(), queue.clone(), &model, surface_config.format)?;
	tracing::info!("Inox2D renderer initialized");
	renderer.resize(surface_config.width, surface_config.height);
	renderer.camera.scale = Vec2::splat(0.15);

	let mut scene_ctrl = ExampleSceneController::new(&renderer.camera, 0.5);

	event_loop.run(move |event, elwt| match event {
		Event::WindowEvent { event, .. } => match event {
			WindowEvent::Resized(size) => {
				surface_config.width = size.width.max(1);
				surface_config.height = size.height.max(1);
				surface.configure(&device, &surface_config);
				renderer.resize(surface_config.width, surface_config.height);
			}
			WindowEvent::CloseRequested => elwt.exit(),
			WindowEvent::KeyboardInput {
				event:
					KeyEvent {
						state: ElementState::Pressed,
						physical_key: PhysicalKey::Code(KeyCode::Escape),
						..
					},
				..
			} => elwt.exit(),
			e => scene_ctrl.interact(&e, &renderer.camera),
		},
		Event::AboutToWait => {
			tracing::debug!("AboutToWait - beginning frame");
			scene_ctrl.update(&mut renderer.camera);

			let puppet = &mut model.puppet;
			puppet.begin_frame();
			let t = scene_ctrl.current_elapsed();
			let _ = puppet
				.param_ctx
				.as_mut()
				.unwrap()
				.set("Head:: Yaw-Pitch", Vec2::new(t.cos(), t.sin()));
			puppet.end_frame(scene_ctrl.dt());

			let frame = match surface.get_current_texture() {
				Ok(f) => f,
				Err(wgpu::SurfaceError::Outdated) => {
					surface.configure(&device, &surface_config);
					surface.get_current_texture().unwrap()
				}
				Err(e) => {
					tracing::error!("Surface error: {:?}", e);
					elwt.exit();
					return;
				}
			};
			tracing::debug!("Acquired frame for drawing");
			let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
			renderer.set_target_view(&view);
			tracing::debug!("Rendering frame");
			renderer.on_begin_draw(puppet);
			renderer.draw(puppet);
			renderer.on_end_draw(puppet);
			frame.present();
			tracing::debug!("Frame presented");
			window.request_redraw();
		}
		_ => {}
	})?;

	Ok(())
}
