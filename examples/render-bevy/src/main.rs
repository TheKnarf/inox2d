use bevy::core_pipeline::core_2d::Camera2d;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input::ButtonInput;
use bevy::prelude::Resource;
use bevy::prelude::*;
use common::scene::ExampleSceneController;
use inox2d::math::camera::Camera as InoxCameraData;
use inox2d_bevy::{Inox2dPlugin, InoxCamera, InoxModelHandle};
use std::env;
use winit::dpi::PhysicalPosition;
use winit::event::{
	DeviceId, ElementState, MouseButton as WinitMouseButton, MouseScrollDelta, TouchPhase, WindowEvent,
};

#[derive(Resource)]
struct SceneController(ExampleSceneController);

#[derive(Resource)]
struct ModelPath(String);

fn setup(mut commands: Commands, assets: Res<AssetServer>, path: Res<ModelPath>) {
	commands.spawn(Camera2d);
	let model: Handle<_> = assets.load(&path.0);
	let camera = InoxCameraData {
		scale: Vec2::splat(0.15),
		..Default::default()
	};
	commands.spawn((
		InoxModelHandle(model),
		InoxCamera(camera.clone()),
		Transform::default(),
		GlobalTransform::default(),
	));
	commands.insert_resource(SceneController(ExampleSceneController::new(&camera, 0.5)));
}

fn control_camera(
	mut scene: ResMut<SceneController>,
	mut camera_q: Query<&mut InoxCamera>,
	buttons: Res<ButtonInput<MouseButton>>,
	mut cursor_evr: EventReader<CursorMoved>,
	mut wheel_evr: EventReader<MouseWheel>,
) {
	let Ok(mut camera) = camera_q.single_mut() else {
		return;
	};

	for ev in cursor_evr.read() {
		let wev = WindowEvent::CursorMoved {
			device_id: unsafe { DeviceId::dummy() },
			position: PhysicalPosition::new(ev.position.x as f64, ev.position.y as f64),
		};
		scene.0.interact(&wev, &camera.0);
	}

	if buttons.just_pressed(MouseButton::Left) {
		let wev = WindowEvent::MouseInput {
			device_id: unsafe { DeviceId::dummy() },
			state: ElementState::Pressed,
			button: WinitMouseButton::Left,
		};
		scene.0.interact(&wev, &camera.0);
	}

	if buttons.just_released(MouseButton::Left) {
		let wev = WindowEvent::MouseInput {
			device_id: unsafe { DeviceId::dummy() },
			state: ElementState::Released,
			button: WinitMouseButton::Left,
		};
		scene.0.interact(&wev, &camera.0);
	}

	for ev in wheel_evr.read() {
		let delta = match ev.unit {
			MouseScrollUnit::Line => MouseScrollDelta::LineDelta(ev.x, ev.y),
			MouseScrollUnit::Pixel => MouseScrollDelta::PixelDelta(PhysicalPosition::new(ev.x as f64, ev.y as f64)),
		};
		let wev = WindowEvent::MouseWheel {
			device_id: unsafe { DeviceId::dummy() },
			delta,
			phase: TouchPhase::Moved,
		};
		scene.0.interact(&wev, &camera.0);
	}

	scene.0.update(&mut camera.0);
}

fn main() {
	let Some(path) = env::args().nth(1) else {
		eprintln!("Usage: render-bevy <MODEL>");
		std::process::exit(1);
	};
	App::new()
		.insert_resource(ModelPath(path))
		.add_plugins(DefaultPlugins)
		.add_plugins(Inox2dPlugin)
		.add_systems(Startup, setup)
		.add_systems(Update, control_camera)
		.run();
}
