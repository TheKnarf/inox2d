use bevy::prelude::*;
use bevy::core_pipeline::core_2d::Camera2d;
use inox2d_bevy::{Inox2dPlugin, InoxModelHandle};
use std::env;

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.spawn(Camera2d);
    let path = env::args().nth(1).expect("Usage: render-bevy <MODEL>");
    let model: Handle<_> = assets.load(path);
    commands.spawn((InoxModelHandle(model), Transform::default(), GlobalTransform::default()));
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inox2dPlugin)
        .add_systems(Startup, setup)
        .run();
}
