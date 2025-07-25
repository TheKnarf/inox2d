use std::path::PathBuf;
use std::{error::Error, fs};

use inox2d::formats::inp::parse_inp;
use inox2d::model::Model;
use inox2d::render::InoxRendererExt;
use inox2d_opengl::OpenglRenderer;

use clap::Parser;
use glam::Vec2;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use winit::event::{ElementState, KeyEvent, WindowEvent};

use common::scene::ExampleSceneController;
use winit::event_loop::EventLoopWindowTarget;
use winit::keyboard::{KeyCode, PhysicalKey};

use app_frame::App;
use winit::window::WindowBuilder;

use crate::app_frame::AppFrame;

mod app_frame;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
	#[arg(help = "Path to the .inp or .inx file.")]
	inp_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
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

	tracing::info!("Setting up windowing and OpenGL");
	let app_frame = AppFrame::init(
		WindowBuilder::new()
			.with_transparent(true)
			.with_resizable(true)
			.with_inner_size(winit::dpi::PhysicalSize::new(600, 800))
			.with_title("Render Inochi2D Puppet (OpenGL)"),
	)?;

	app_frame.run(Inox2dOpenglExampleApp::new(model))?;

	Ok(())
}

struct Inox2dOpenglExampleApp {
	on_window: Option<(OpenglRenderer, ExampleSceneController)>,
	model: Model,
	width: u32,
	height: u32,
}

impl Inox2dOpenglExampleApp {
	pub fn new(model: Model) -> Self {
		Self {
			on_window: None,
			model,
			width: 0,
			height: 0,
		}
	}
}

impl App for Inox2dOpenglExampleApp {
	fn resume_window(&mut self, gl: glow::Context) {
		match OpenglRenderer::new(gl, &self.model) {
			Ok(mut renderer) => {
				tracing::info!("Initializing Inox2D renderer");
				renderer.resize(self.width, self.height);
				renderer.camera.scale = Vec2::splat(0.15);
				tracing::info!("Inox2D renderer initialized");

				let scene_ctrl = ExampleSceneController::new(&renderer.camera, 0.5);
				self.on_window = Some((renderer, scene_ctrl));
			}
			Err(e) => {
				tracing::error!("{}", e);
				self.on_window = None;
			}
		}
	}

	fn resize(&mut self, width: i32, height: i32) {
		self.width = width as u32;
		self.height = height as u32;

		if let Some((renderer, _)) = &mut self.on_window {
			renderer.resize(self.width, self.height);
		}
	}

	fn draw(&mut self) {
		let Some((renderer, scene_ctrl)) = &mut self.on_window else {
			return;
		};

		tracing::debug!("Redrawingggggg");
		scene_ctrl.update(&mut renderer.camera);

		renderer.clear();

		let puppet = &mut self.model.puppet;
		puppet.begin_frame();
		let t = scene_ctrl.current_elapsed();
		let _ = puppet
			.param_ctx
			.as_mut()
			.unwrap()
			.set("Head:: Yaw-Pitch", Vec2::new(t.cos(), t.sin()));
		// Actually, not providing 0 for the first frame will not create too big a problem.
		// Just that physics simulation will run for the provided time, which may be big and causes a startup delay.
		puppet.end_frame(scene_ctrl.dt());

		renderer.on_begin_draw(puppet);
		renderer.draw(puppet);
		renderer.on_end_draw(puppet);
	}

	fn handle_window_event(&mut self, event: WindowEvent, elwt: &EventLoopWindowTarget<()>) {
		match event {
			WindowEvent::CloseRequested => elwt.exit(),
			WindowEvent::KeyboardInput {
				event:
					KeyEvent {
						state: ElementState::Pressed,
						physical_key: PhysicalKey::Code(KeyCode::Escape),
						..
					},
				..
			} => {
				tracing::info!("There is an Escape D:");
				elwt.exit();
			}
			event => {
				if let Some((renderer, scene_ctrl)) = &mut self.on_window {
					scene_ctrl.interact(&event, &renderer.camera)
				}
			}
		}
	}
}
