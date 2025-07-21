use bevy::core_pipeline::core_2d::Camera2d;
use bevy::prelude::*;
use inox2d::math::camera::Camera as InoxCameraData;
use inox2d_bevy::{Inox2dPlugin, InoxCamera, InoxModelHandle};
use std::env;

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
	commands.spawn(Camera2d);
	let path = env::args().nth(1).expect("Usage: render-bevy <MODEL>");
	let model: Handle<_> = assets.load(path);
        commands.spawn((
                InoxModelHandle(model),
                InoxCamera(InoxCameraData {
                        scale: Vec2::splat(0.15),
                        ..Default::default()
                }),
                Transform::default(),
                GlobalTransform::default(),
        ));
}

fn main() {
	App::new()
		.add_plugins(DefaultPlugins)
		.add_plugins(Inox2dPlugin)
		.add_systems(Startup, setup)
		.run();
}
