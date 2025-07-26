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
}
