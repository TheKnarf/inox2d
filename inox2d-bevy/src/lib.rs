//! Bevy plugin for rendering Inox2D puppets.
//!
//! Attach [`InoxModelHandle`], [`InoxWgpuRenderer`] and [`InoxCamera`] to an
//! entity to display a model. Modify the fields of [`InoxCamera`] to move,
//! scale or rotate the view. The [`sync_inox_camera`] system copies these
//! values into the renderer every frame.

use bevy::asset::AssetLoader;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::ViewTarget;
use bevy::render::{MainWorld, Render, RenderApp, RenderSet};
use futures_lite::future::block_on;
use inox2d::formats::inp::{parse_inp, ParseInpError};
use inox2d::math::camera::Camera;
use inox2d::model::Model;
use inox2d::node::components::BlendMode;
use inox2d::render::InoxRendererExt;
use inox2d_wgpu::WgpuRenderer;
use tracing::error;

#[derive(Debug, thiserror::Error)]
pub enum InoxAssetError {
	#[error("{0}")]
	Parse(#[from] ParseInpError),
	#[error("I/O error while reading asset: {0}")]
	Io(#[from] bevy::asset::io::AssetReaderError),
}

#[derive(Asset, TypePath)]
pub struct InoxAsset(pub Model);

#[derive(Default)]
pub struct InoxAssetLoader;

impl AssetLoader for InoxAssetLoader {
	type Asset = InoxAsset;
	type Settings = ();
	type Error = InoxAssetError;

	#[allow(refining_impl_trait)]
	fn load(
		&self,
		reader: &mut dyn bevy::asset::io::Reader,
		_: &Self::Settings,
		_: &mut bevy::asset::LoadContext,
	) -> bevy::tasks::BoxedFuture<'static, Result<Self::Asset, Self::Error>> {
		let mut buf = Vec::new();
		let res: Result<InoxAsset, InoxAssetError> = (|| {
			block_on(reader.read_to_end(&mut buf)).map_err(|e| InoxAssetError::Io(e.into()))?;
			let mut model = parse_inp(buf.as_slice())?;
			model.puppet.init_transforms();
			model.puppet.init_rendering();
			model.puppet.init_params();
			model.puppet.init_physics();
			Ok(InoxAsset(model))
		})();
		Box::pin(async move { res })
	}

	fn extensions(&self) -> &[&str] {
		&["inp", "inx"]
	}
}

pub struct Inox2dPlugin;

impl Plugin for Inox2dPlugin {
	fn build(&self, app: &mut App) {
		app.init_asset_loader::<InoxAssetLoader>()
			.init_asset::<InoxAsset>()
			.add_event::<RendererInitFailed>()
			.add_systems(Update, (update_puppets, sync_inox_camera, sync_inox_render_config));

		if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
			render_app.add_systems(Render, draw_puppets.in_set(RenderSet::Render).after(RenderSet::Render));
		}
	}
}
#[derive(Component)]
pub struct InoxModelHandle(pub Handle<InoxAsset>);

#[derive(Component)]
pub struct InoxWgpuRenderer(pub WgpuRenderer);

#[derive(Event, Debug, Clone)]
pub struct RendererInitFailed {
	pub entity: Entity,
	pub error: String,
}

#[derive(Component, Debug)]
pub struct InoxRendererError(pub String);

#[derive(Component, Clone)]
pub struct InoxRenderConfig {
	pub blend_mode: BlendMode,
	pub tint: Vec3,
	pub emission_strength: f32,
	pub mask_threshold: f32,
}

impl Default for InoxRenderConfig {
	fn default() -> Self {
		Self {
			blend_mode: BlendMode::Normal,
			tint: Vec3::ONE,
			emission_strength: 1.0,
			mask_threshold: 0.5,
		}
	}
}

/// Camera component controlling how Inox2D content is viewed.
///
/// Adjust [`Camera::position`], [`Camera::scale`] and [`Camera::rotation`] to
/// move and zoom the rendered model.
#[derive(Component, Clone)]
pub struct InoxCamera(pub Camera);

impl Default for InoxCamera {
	fn default() -> Self {
		Self(Camera::default())
	}
}

pub fn sync_inox_camera(mut query: Query<(&InoxCamera, &mut InoxWgpuRenderer)>) {
	for (camera, mut renderer) in &mut query {
		renderer.0.camera = camera.0.clone();
	}
}

pub fn sync_inox_render_config(mut query: Query<(&InoxRenderConfig, &mut InoxWgpuRenderer)>) {
	for (config, mut renderer) in &mut query {
		renderer.0.blend_mode = config.blend_mode;
		renderer.0.tint = config.tint;
		renderer.0.emission_strength = config.emission_strength;
		renderer.0.mask_threshold = config.mask_threshold;
	}
}

pub fn update_puppets(
	time: Res<Time>,
	mut assets: ResMut<Assets<InoxAsset>>,
	render_device: Option<Res<RenderDevice>>,
	render_queue: Option<Res<RenderQueue>>,
	mut commands: Commands,
	mut error_events: EventWriter<RendererInitFailed>,
	mut query: Query<(Entity, &InoxModelHandle, &mut Transform, Option<&mut InoxWgpuRenderer>)>,
) {
	for (entity, handle, mut transform, renderer) in &mut query {
		if let Some(model) = assets.get_mut(&handle.0) {
			if renderer.is_none() {
				if let (Some(device), Some(queue)) = (render_device.as_ref(), render_queue.as_ref()) {
					let wgpu_queue = queue.0.as_ref().clone().into_inner();
					match WgpuRenderer::new(device.wgpu_device().clone(), wgpu_queue, &model.0) {
						Ok(r) => {
							commands
								.entity(entity)
								.insert(InoxWgpuRenderer(r))
								.remove::<InoxRendererError>();
						}
						Err(e) => {
							error!("failed to create WgpuRenderer: {}", e);
							error_events.send(RendererInitFailed {
								entity,
								error: e.to_string(),
							});
							commands.entity(entity).insert(InoxRendererError(e.to_string()));
						}
					}
				}
			}

			// simple rotation over time
			transform.rotation = Quat::from_rotation_z(time.elapsed_secs());

			let puppet = &mut model.0.puppet;
			puppet.begin_frame();
			puppet.end_frame(time.delta_secs());
		}
	}
}

pub fn draw_puppets(mut main_world: ResMut<MainWorld>, targets: Query<&ViewTarget>) {
	let Ok(view_target) = targets.get_single() else {
		return;
	};
	let width = view_target.main_texture().width();
	let height = view_target.main_texture().height();

	main_world.resource_scope(|world, mut assets: Mut<Assets<InoxAsset>>| {
		let mut query = world.query::<(&InoxModelHandle, &mut InoxWgpuRenderer)>();
		for (handle, mut renderer) in query.iter_mut(world) {
			if let Some(model) = assets.get(&handle.0) {
				if renderer.0.viewport.x != width || renderer.0.viewport.y != height {
					renderer.0.resize(width, height);
				}
				renderer.0.set_target_view(view_target.out_texture());
				renderer.0.on_begin_draw(&model.0.puppet);
				renderer.0.draw(&model.0.puppet);
				renderer.0.on_end_draw(&model.0.puppet);
			}
		}
	});
}
