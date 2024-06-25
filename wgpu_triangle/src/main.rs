use std::sync::Arc;
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
  window::{Window, WindowId},
};
use pollster::FutureExt as _;

type Size = winit::dpi::PhysicalSize<u32>;

struct State {
  window: Arc<Window>,
  surface: wgpu::Surface<'static>,
  instance: wgpu::Instance,
  config: wgpu::SurfaceConfiguration,
  queue: wgpu::Queue,
  device: wgpu::Device,
  size: Size,
}

#[derive(Default)]
struct App {
  state: Option<State>,
}

impl App {
  pub fn resize(&mut self, new_size: Size) {
    if new_size.width == 0 || new_size.height == 0 {
      return;
    }
    if let Some(state) = &mut self.state {
      state.size = new_size;
      state.config.width = state.size.width;
      state.config.height = state.size.height;
      state.surface.configure(&state.device, &state.config);
    }
  }
  pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    let state = match &mut self.state {
      Some(state) => state,
      None => {
        return Ok(());
      }
    };
    let output = state.surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let enc_desc = wgpu::CommandEncoderDescriptor {
      label: Some("render encoder"),
    };
    let mut encoder = state.device.create_command_encoder(&enc_desc);
    {
      let color_attach = wgpu::RenderPassColorAttachment {
        view: &view,
        resolve_target: None,
        ops: wgpu::Operations {
          load: wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.1, g: 0.1, b: 0.1, a: 1.0,
          }),
          store: wgpu::StoreOp::Store,
        }
      };
      let rp_desc = wgpu::RenderPassDescriptor {
        label: Some("render pass"),
        color_attachments: &[Some(color_attach)],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
      };
      let render_pass = encoder.begin_render_pass(&rp_desc);
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    Ok(())
  }
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
      device,
    });
  }

  fn window_event(
    &mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent,
  ) {
    let state = match &self.state {
      Some(state) => state,
      None => return,
    };
    // not our window
    if state.window.id() != id {
      return;
    }
    let size = state.size;

    match event {
      WindowEvent::CloseRequested => {
        event_loop.exit();
      },
      WindowEvent::RedrawRequested => {
        state.window.request_redraw();
        let res = self.render();
        match res {
          Ok(_) => {},
          // surface needs to be reconfigured
          Err(wgpu::SurfaceError::Lost) => {
            self.resize(size);
          },
          // no memory, abort
          Err(wgpu::SurfaceError::OutOfMemory) => {
            event_loop.exit();
          },
          // ignore other errors
          Err(e) => {
            eprintln!("{:?}", e);
          },
        };
      },
      WindowEvent::Resized(size) => {
        self.resize(size);
      },
      _ => {}
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
