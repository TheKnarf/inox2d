use bevy::asset::AssetLoader;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use futures_lite::future::block_on;
use inox2d::formats::inp::{parse_inp, ParseInpError};
use inox2d::model::Model;
use inox2d::render::InoxRendererExt;
use inox2d_wgpu::WgpuRenderer;

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
			.add_systems(Update, (update_puppets, draw_puppets));
	}
}
#[derive(Component)]
pub struct InoxModelHandle(pub Handle<InoxAsset>);

#[derive(Component)]
pub struct InoxWgpuRenderer(pub WgpuRenderer);

pub fn update_puppets(
        time: Res<Time>,
        mut assets: ResMut<Assets<InoxAsset>>,
        render_device: Option<Res<RenderDevice>>,
        render_queue: Option<Res<RenderQueue>>,
        mut commands: Commands,
        mut query: Query<(Entity, &InoxModelHandle, &mut Transform, Option<&mut InoxWgpuRenderer>)>,
) {
        for (entity, handle, mut transform, renderer) in &mut query {
                if let Some(model) = assets.get_mut(&handle.0) {
                        if renderer.is_none() {
                                if let (Some(device), Some(queue)) = (render_device.as_ref(), render_queue.as_ref()) {
                                        let wgpu_queue = queue.0.as_ref().clone().into_inner();
                                        if let Ok(r) = WgpuRenderer::new(device.wgpu_device().clone(), wgpu_queue, &model.0) {
                                                commands.entity(entity).insert(InoxWgpuRenderer(r));
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

pub fn draw_puppets(assets: Res<Assets<InoxAsset>>, query: Query<(&InoxModelHandle, &InoxWgpuRenderer)>) {
	for (handle, renderer) in &query {
		if let Some(model) = assets.get(&handle.0) {
			renderer.0.draw(&model.0.puppet);
		}
	}
}
