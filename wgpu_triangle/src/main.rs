use std::sync::Arc;
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
  window::{Window, WindowId},
  dpi::PhysicalSize,
};
use pollster::FutureExt as _;

struct State {
  window: Arc<Window>,
  surface: wgpu::Surface<'static>,
  instance: wgpu::Instance,
  config: wgpu::SurfaceConfiguration,
  queue: wgpu::Queue,
  size: PhysicalSize<u32>,
}

#[derive(Default)]
struct App {
  state: Option<State>,
}

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      backends: wgpu::Backends::PRIMARY,
      ..Default::default()
    });
    let window = Arc::new(
      event_loop.create_window(Window::default_attributes()).unwrap()
    );
    let size = window.inner_size();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
      .enumerate_adapters(wgpu::Backends::all())
      .into_iter()
      .filter(|adapter| {
        adapter.is_surface_supported(&surface)
      })
      .next().unwrap();
    let (device, queue) = adapter.request_device(
      &wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        label: None
      },
      None
    ).block_on().unwrap();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats.iter()
      .find(|f| f.is_srgb())
      .copied()
      .unwrap_or(surface_caps.formats[0]);
    if !surface_caps.alpha_modes.iter().any(|&m| m == wgpu::CompositeAlphaMode::Opaque) {
      panic!("Opaque alpha blend required");
    }
    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format: surface_format,
      width: size.width,
      height: size.height,
      // todo: make selectable?
      present_mode: surface_caps.present_modes[0],
      alpha_mode: wgpu::CompositeAlphaMode::Opaque,
      view_formats: vec![],
      desired_maximum_frame_latency: 2,
    };

    self.state = Some(State {
      size,
      instance,
      queue,
      config,
      surface,
      window,
    });
  }

  fn window_event(
    &mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent,
  ) {
    if let Some(state) = &self.state {
      // not our window
      if state.window.id() != id {
        return;
      }
      match event {
        WindowEvent::CloseRequested => {
          event_loop.exit();
        },
        WindowEvent::RedrawRequested => {
          state.window.request_redraw();
        },
        _ => {}
      }
    }
    else {
      // no state, i.e. not initialized
      return;
    }
  }
}

pub fn main() {
  env_logger::init();
  let event_loop = EventLoop::new().unwrap();
  event_loop.set_control_flow(ControlFlow::Poll);
  let mut app = App::default();
  event_loop.run_app(&mut app).unwrap();
}
