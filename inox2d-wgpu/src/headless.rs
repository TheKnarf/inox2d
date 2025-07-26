#![cfg(feature = "headless")]

/// Initialize a headless [`wgpu`] context for tests.
///
/// Creates a [`wgpu::Instance`], obtains a headless [`wgpu::Surface`],
/// requests an adapter and device and returns the surface together with
/// the created device and queue.
pub async fn init_headless_wgpu(
    size: wgpu::Extent3d,
) -> Result<(wgpu::Surface<'static>, wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    use winit::event_loop::EventLoopBuilder;
    let mut builder = EventLoopBuilder::new();
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        builder.with_any_thread(true);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::x11::EventLoopBuilderExtX11;
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    let event_loop = builder.build().expect("Failed to create event loop");
    let window = winit::window::WindowBuilder::new()
        .with_visible(false)
        .with_inner_size(winit::dpi::PhysicalSize::new(size.width, size.height))
        .build(&event_loop)
        .expect("Failed to create window");
    let window = Box::leak(Box::new(window));

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(window).expect("Failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Adapter request failed");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    Ok((surface, device, queue))
}

#[cfg(all(test, feature = "headless"))]
mod tests {
    use super::*;
    use crate::WgpuRenderer;
    use futures::channel::oneshot;
    use image::{codecs::png::PngEncoder, ColorType, ImageBuffer, ImageEncoder, Rgba};
    use inox2d::{
        model::{Model, ModelTexture},
        puppet::Puppet,
        render::InoxRendererExt,
    };
    use std::sync::Arc;

    #[test]
    fn init_headless_wgpu_creates_device() {
        #[cfg(all(unix, not(target_os = "macos")))]
        if std::env::var("DISPLAY").is_err() && std::env::var("WAYLAND_DISPLAY").is_err() {
            eprintln!("Skipping test: no display server available");
            return;
        }

        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let result = pollster::block_on(init_headless_wgpu(size));
        assert!(result.is_ok(), "headless initialization failed: {:?}", result);

        let (_surface, device, queue) = result.unwrap();

        // simple sanity check that the device and queue can create and submit
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.submit(std::iter::empty());
        drop(buffer);
    }

    #[test]
    fn render_non_zero_alpha() {
        #[cfg(all(unix, not(target_os = "macos")))]
        if std::env::var("DISPLAY").is_err() && std::env::var("WAYLAND_DISPLAY").is_err() {
            eprintln!("Skipping test: no display server available");
            return;
        }

        let size = wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };

        let (_surface, device, queue) =
            pollster::block_on(init_headless_wgpu(size)).expect("init headless");

        let img = ImageBuffer::<Rgba<u8>, _>::from_fn(2, 2, |_, _| Rgba([255, 0, 0, 255]));
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(img.as_raw(), 2, 2, ColorType::Rgba8.into())
            .unwrap();
        let tex = ModelTexture {
            format: image::ImageFormat::Png,
            data: Arc::from(png.into_boxed_slice()),
        };

        let puppet_json = r#"{
            "meta": { "version": "1" },
            "physics": { "pixelsPerMeter": 100.0, "gravity": 9.8 },
            "param": [],
            "nodes": {
                "uuid": 0,
                "name": "root",
                "enabled": true,
                "zsort": 0.0,
                "transform": {
                    "trans": [0.0,0.0,0.0],
                    "rot": [0.0,0.0,0.0],
                    "scale": [1.0,1.0],
                    "pixel_snap": false
                },
                "lockToRoot": false,
                "type": "Part",
                "blend_mode": "Normal",
                "textures": [0],
                "mesh": {
                    "verts": [0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0],
                    "uvs":   [0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0],
                    "indices": [0,1,2,0,2,3],
                    "origin": [0.0,0.0]
                },
                "children": []
            }
        }"#;
        let payload = json::parse(puppet_json).unwrap();
        let mut puppet = Puppet::new_from_json(&payload).unwrap();
        puppet.init_transforms();
        puppet.init_rendering();

        let model = Model {
            puppet,
            textures: vec![tex],
            vendors: Vec::new(),
        };

        let mut renderer =
            WgpuRenderer::new(device.clone(), queue.clone(), &model, wgpu::TextureFormat::Rgba8UnormSrgb)
                .unwrap();
        renderer.resize(size.width, size.height);

        let puppet = &model.puppet;
        renderer.on_begin_draw(puppet);
        renderer.draw(puppet);
        renderer.on_end_draw(puppet);
        device.poll(wgpu::Maintain::Wait);

        let bytes_per_row = ((4 * size.width + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * size.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.copy_target_to_buffer(&mut encoder, &renderer.offscreen_texture, &buffer);
        queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let has_alpha = data.chunks(4).any(|p| p[3] != 0);
        drop(data);
        buffer.unmap();

        assert!(has_alpha, "Rendered image should have non-zero alpha pixel");
    }

    #[test]
    fn render_simple_model_to_texture() {
        #[cfg(all(unix, not(target_os = "macos")))]
        if std::env::var("DISPLAY").is_err() && std::env::var("WAYLAND_DISPLAY").is_err() {
            eprintln!("Skipping test: no display server available");
            return;
        }

        let size = wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        };

        let (_surface, device, queue) =
            pollster::block_on(init_headless_wgpu(size)).expect("init headless");

        let img = ImageBuffer::<Rgba<u8>, _>::from_fn(1, 1, |_, _| Rgba([0, 255, 0, 255]));
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(img.as_raw(), 1, 1, ColorType::Rgba8.into())
            .unwrap();
        let tex = ModelTexture {
            format: image::ImageFormat::Png,
            data: Arc::from(png.into_boxed_slice()),
        };

        let puppet_json = r#"{
            "meta": { "version": "1" },
            "physics": { "pixelsPerMeter": 100.0, "gravity": 9.8 },
            "param": [],
            "nodes": {
                "uuid": 0,
                "name": "root",
                "enabled": true,
                "zsort": 0.0,
                "transform": {
                    "trans": [0.0,0.0,0.0],
                    "rot": [0.0,0.0,0.0],
                    "scale": [1.0,1.0],
                    "pixel_snap": false
                },
                "lockToRoot": false,
                "type": "Part",
                "blend_mode": "Normal",
                "textures": [0],
                "mesh": {
                    "verts": [0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0],
                    "uvs":   [0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0],
                    "indices": [0,1,2,0,2,3],
                    "origin": [0.0,0.0]
                },
                "children": []
            }
        }"#;
        let payload = json::parse(puppet_json).unwrap();
        let mut puppet = Puppet::new_from_json(&payload).unwrap();
        puppet.init_transforms();
        puppet.init_rendering();

        let model = Model {
            puppet,
            textures: vec![tex],
            vendors: Vec::new(),
        };

        let mut renderer =
            WgpuRenderer::new(device.clone(), queue.clone(), &model, wgpu::TextureFormat::Rgba8UnormSrgb)
                .unwrap();
        renderer.resize(size.width, size.height);

        let puppet = &model.puppet;
        renderer.on_begin_draw(puppet);
        renderer.draw(puppet);
        renderer.on_end_draw(puppet);
        device.poll(wgpu::Maintain::Wait);

        let bytes_per_row = ((4 * size.width + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * size.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.copy_target_to_buffer(&mut encoder, &renderer.offscreen_texture, &buffer);
        queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let has_alpha = data.chunks(4).any(|p| p[3] != 0);
        drop(data);
        buffer.unmap();

        assert!(has_alpha, "Rendered image should have non-zero alpha pixel");
    }
}
