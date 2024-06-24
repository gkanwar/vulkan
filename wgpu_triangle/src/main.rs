use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
  window::{Window, WindowId},
};

#[derive(Default)]
struct App {
  window: Option<Window>,
}

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
    self.window = Some(event_loop.create_window(Window::default_attributes()).unwrap());
  }

  fn window_event(
    &mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent,
  ) {
    if let Some(window) = &self.window {
      // not our window
      if window.id() != id {
        return;
      }
    }
    else {
      // no window
      return;
    }
    match event {
      WindowEvent::CloseRequested => {
        event_loop.exit();
      },
      WindowEvent::RedrawRequested => {
        self.window.as_ref().unwrap().request_redraw();
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
