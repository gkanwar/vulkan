use std::sync::Arc;
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
  window::{Window, WindowId},
};
use wgpu::util::DeviceExt;
use pollster::FutureExt as _;
use bytemuck::{
  Pod, Zeroable,
};
use glam::{
  Vec3, Mat4, Quat,
};

type Size = winit::dpi::PhysicalSize<u32>;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VertexCore {
  pos: [f32; 3],
}
impl VertexCore {
  pub fn new(pos: [f32; 3]) -> Self {
    Self { pos }
  }
}
impl From<[f32; 3]> for VertexCore {
  fn from(pos: [f32; 3]) -> Self {
    Self { pos }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VertexAux {
  color: [f32; 3],
}
impl VertexAux {
  pub fn new(color: [f32; 3]) -> Self {
    Self { color }
  }
}

const VERT_CORE_LAYOUT: wgpu::VertexBufferLayout  =
  wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<VertexCore>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
  };
const VERT_AUX_LAYOUT: wgpu::VertexBufferLayout =
  wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<VertexAux>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![1 => Float32x3],
  };

struct VertexBufferLayouts {
  xs_layout: wgpu::VertexBufferLayout<'static>,
  colors_layout: wgpu::VertexBufferLayout<'static>,
}


#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct VertPushConstants {
  model: Mat4,
  view: Mat4,
  proj: Mat4,
}

const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
  1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 0.5, 0.5,
  0.0, 0.0, 0.0, 1.0,
]);

struct Camera {
  // view
  eye: Vec3,
  target: Vec3,
  up: Vec3,
  // proj
  fovy: f32,
  aspect: f32,
  znear: f32,
  zfar: f32,
}
impl Camera {
  fn get_view(&self) -> Mat4 {
    Mat4::look_at_rh(self.eye, self.target, self.up)
  }
  fn get_proj(&self) -> Mat4 {
    OPENGL_TO_WGPU_MATRIX * Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
  }
  // set aspect to match viewport size
  fn set_aspect(&mut self, size: Size) {
    self.aspect = (size.width as f32) / (size.height as f32);
  }
}
impl Default for Camera {
  fn default() -> Self {
    let eye = Vec3::new(0.0, 2.0, 2.0);
    let target = Vec3::new(0.0, 0.0, 0.0);
    let up = Vec3::new(0.0, 0.0, 1.0);
    let fovy = (45.0_f32).to_radians();
    let aspect = 1.0;
    let znear = 0.1;
    let zfar = 10.0;
    Self {
      eye, target, up,
      fovy, aspect, znear, zfar,
    }
  }
}

struct Mesh {
  xs: Vec<VertexCore>,
  colors: Vec<VertexAux>,
  inds: Vec<u32>,

  shift: Vec3,
  rot: Quat,
  scale: Vec3,

  xs_buffer: Option<wgpu::Buffer>,
  colors_buffer: Option<wgpu::Buffer>,
  inds_buffer: Option<wgpu::Buffer>,
}

impl Mesh {
  fn vert_buffer_layouts() -> VertexBufferLayouts {
    VertexBufferLayouts {
      xs_layout: VERT_CORE_LAYOUT,
      colors_layout: VERT_AUX_LAYOUT,
    }
  }

  fn inds_format() -> wgpu::IndexFormat {
    wgpu::IndexFormat::Uint32
  }

  fn get_transform(&self) -> Mat4 {
    Mat4::from_scale_rotation_translation(self.scale, self.rot, self.shift)
  }
}

impl Default for Mesh {
  fn default() -> Self {
    Self {
      xs: vec![],
      colors: vec![],
      inds: vec![],
      shift: Vec3::ZERO,
      rot: Quat::IDENTITY,
      scale: Vec3::ONE,
      xs_buffer: None,
      colors_buffer: None,
      inds_buffer: None,
    }
  }
}

struct GraphicsState {
  window: Arc<Window>,
  surface: wgpu::Surface<'static>,
  instance: wgpu::Instance,
  config: wgpu::SurfaceConfiguration,
  queue: wgpu::Queue,
  device: wgpu::Device,
  pipeline: wgpu::RenderPipeline,
  size: Size,
}

#[derive(Default)]
struct App {
  state: Option<GraphicsState>,
  meshes: Vec<Mesh>,
  camera: Camera,
  start_time: Option<std::time::Instant>,
}

impl App {
  pub fn start(&mut self) {
    self.start_time = Some(std::time::Instant::now());
  }

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
    let mut push_consts = VertPushConstants {
      model: Mat4::IDENTITY,
      view: self.camera.get_view(),
      proj: self.camera.get_proj(),
    };
    let mut encoder = state.device.create_command_encoder(&enc_desc);
    {
      let color_attaches = [Some(wgpu::RenderPassColorAttachment {
        view: &view,
        resolve_target: None,
        ops: wgpu::Operations {
          load: wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.1, g: 0.1, b: 0.1, a: 1.0,
          }),
          store: wgpu::StoreOp::Store,
        }
      })];
      let rp_desc = wgpu::RenderPassDescriptor {
        label: Some("render pass"),
        color_attachments: &color_attaches,
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
      };
      let mut render_pass = encoder.begin_render_pass(&rp_desc);
      render_pass.set_pipeline(&state.pipeline);
      self.meshes.iter().for_each(|mesh| {
        render_pass.set_vertex_buffer(0, mesh.xs_buffer.as_ref().unwrap().slice(..));
        render_pass.set_vertex_buffer(1, mesh.colors_buffer.as_ref().unwrap().slice(..));
        render_pass.set_index_buffer(mesh.inds_buffer.as_ref().unwrap().slice(..), Mesh::inds_format());
        push_consts.model = mesh.get_transform();
        render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&push_consts));
        let inds = 0..(mesh.inds.len() as u32);
        let vert_off = 0;
        let insts = 0..1;
        render_pass.draw_indexed(inds, vert_off, insts);
      });
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    Ok(())
  }

  fn update(&mut self) {
    let time = self.start_time.unwrap().elapsed().as_secs_f32();
    self.meshes.iter_mut().for_each(|mesh| {
      let theta = time * (90.0_f32).to_radians();
      mesh.rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), theta);
    });
  }

  fn init_mesh_buffers(&mut self) {
    let state = &mut self.state.as_mut().unwrap();
    self.meshes.iter_mut().for_each(|mesh| {
      let buffer_desc = wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&mesh.xs),
        usage: wgpu::BufferUsages::VERTEX,
      };
      mesh.xs_buffer = Some(state.device.create_buffer_init(&buffer_desc));
      let buffer_desc = wgpu::util::BufferInitDescriptor {
        label: Some("colors buffer"),
        contents: bytemuck::cast_slice(&mesh.colors),
        usage: wgpu::BufferUsages::VERTEX,
      };
      mesh.colors_buffer = Some(state.device.create_buffer_init(&buffer_desc));
      let buffer_desc = wgpu::util::BufferInitDescriptor {
        label: Some("inds buffer"),
        contents: bytemuck::cast_slice(&mesh.inds),
        usage: wgpu::BufferUsages::INDEX,
      };
      mesh.inds_buffer = Some(state.device.create_buffer_init(&buffer_desc));
    });
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
        adapter.is_surface_supported(&surface) &&
          adapter.features().contains(wgpu::Features::PUSH_CONSTANTS)
      })
      .next().unwrap();
    let (device, queue) = adapter.request_device(
      &wgpu::DeviceDescriptor {
        required_features: wgpu::Features::PUSH_CONSTANTS,
        required_limits: wgpu::Limits {
          max_push_constant_size: 256,
          .. Default::default()
        },
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

    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));
    let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
      label: Some("render pipeline layout"),
      bind_group_layouts: &[],
      push_constant_ranges: &[wgpu::PushConstantRange {
        range: 0..(std::mem::size_of::<VertPushConstants>() as u32),
        stages: wgpu::ShaderStages::VERTEX,
      }],
    };
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);
    let frag_targets = [Some(wgpu::ColorTargetState {
      format: config.format,
      blend: Some(wgpu::BlendState::REPLACE),
      write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline_desc = wgpu::RenderPipelineDescriptor {
      label: Some("render pipeline"),
      layout: Some(&pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: "vs_main",
        buffers: &[
          Mesh::vert_buffer_layouts().xs_layout,
          Mesh::vert_buffer_layouts().colors_layout,
        ],
        // todo: push constants here?
        compilation_options: wgpu::PipelineCompilationOptions::default(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: "fs_main",
        targets: &frag_targets,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      // todo
      depth_stencil: None,
      multisample: wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
      },
      multiview: None,
    };
    let pipeline = device.create_render_pipeline(&pipeline_desc);

    // request initial draw
    window.request_redraw();

    self.state = Some(GraphicsState {
      size,
      instance,
      queue,
      config,
      surface,
      window,
      device,
      pipeline,
    });

    // setup meshes
    self.init_mesh_buffers();

    // setup camera
    self.camera.set_aspect(size);
  }

  fn window_event(
    &mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent,
  ) {
    println!("window_event {:?}", event);
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
        self.update();
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
  app.meshes.push(Mesh {
    xs: [
      [0.5, -0.5, 0.0],
      [-0.5, 0.5, 0.0],
      [0.5, 0.5, 0.0],
      [-0.5, -0.5, 0.0],
    ].into_iter().map(VertexCore::new).collect(),
    colors: [
      [1.0, 1.0, 1.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [1.0, 0.0, 0.0],
    ].into_iter().map(VertexAux::new).collect(),
    inds: vec![
      1, 0, 2,
      0, 1, 3,
    ],
    .. Default::default()
  });
  app.camera = Default::default();
  app.start();
  event_loop.run_app(&mut app).unwrap();
}
