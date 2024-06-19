#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

// clip depth to [0,1] range
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <optional>
#include <set>
#include <vector>

const std::vector<const char*> g_validation_layers = {
  "VK_LAYER_KHRONOS_validation",
};
#ifndef NDEBUG
const bool g_enable_validation_layers = true;
#else
const bool g_enable_validation_layers = false;
#endif

const std::vector<const char*> g_device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool allAvailable() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

// TODO:
struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR caps;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> modes;
};

class Application {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  void initWindow() {
    glfwInit();
    // no OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // non-resizable
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(800, 600, "Hello triangle", nullptr, nullptr);
  }
  
  void initVulkan() {
    createVulkanInstance();
    createSurface();
    selectVulkanPhysicalDevice();
    createVulkanLogicalDevice();
  }

  void createVulkanInstance() {
    if (g_enable_validation_layers && !checkValidationLayerSupport()) {
      throw std::runtime_error("validation layers enabled but not supported\n");
    }
    
    vk::ApplicationInfo app_info = {};
    app_info.sType = vk::StructureType::eApplicationInfo;
    app_info.pApplicationName = "Hello triangle";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "None";
    app_info.engineVersion = VK_MAKE_VERSION(0, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    vk::InstanceCreateInfo inst_info = {};
    inst_info.sType = vk::StructureType::eInstanceCreateInfo;
    inst_info.pApplicationInfo = &app_info;

    uint32_t glfw_n_extension = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_n_extension);
    inst_info.enabledExtensionCount = glfw_n_extension;
    inst_info.ppEnabledExtensionNames = glfw_extensions;
    inst_info.enabledLayerCount = 0;

    auto res = vk::createInstance(&inst_info, nullptr, &m_instance);
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("createInstance");
    }
  }

  void createSurface() {
    auto res = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    if (res != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface");
    }
  }

  void selectVulkanPhysicalDevice() {
    uint32_t n_device = 0;
    auto res = m_instance.enumeratePhysicalDevices(&n_device, nullptr);
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("");
    }
    if (n_device == 0) {
      throw std::runtime_error("no supported Vulkan devices available");
    }
    std::vector<vk::PhysicalDevice> devices(n_device);
    res = m_instance.enumeratePhysicalDevices(&n_device, devices.data());
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("");
    }
    for (const auto& device : devices) {
      if (isDeviceSuitable(device)) {
        m_phys_device = device;
        break;
      }
    }
    if (m_phys_device == VK_NULL_HANDLE) {
      throw std::runtime_error("no supported Vulkan devices available");
    }
    else {
      vk::PhysicalDeviceProperties props;
      m_phys_device.getProperties(&props);
      std::cout << "Selected GPU: " << props.deviceName << "\n";
    }
  }

  void createVulkanLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(m_phys_device);
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    std::set<uint32_t> unique_queue_families = {
      indices.graphics_family.value(),
      indices.present_family.value(),
    };

    float priority = 1.0f;

    for (uint32_t family : unique_queue_families) {
      vk::DeviceQueueCreateInfo queue_info = {};
      queue_info.sType = vk::StructureType::eDeviceQueueCreateInfo;
      queue_info.queueFamilyIndex = family;
      queue_info.queueCount = 1;
      queue_info.pQueuePriorities = &priority;
      queue_infos.push_back(queue_info);
    }

    vk::PhysicalDeviceFeatures device_features = {};

    vk::DeviceCreateInfo device_info = {};
    device_info.sType = vk::StructureType::eDeviceCreateInfo; 
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.queueCreateInfoCount = queue_infos.size();
    device_info.pEnabledFeatures = &device_features;
    device_info.enabledExtensionCount = g_device_extensions.size();
    device_info.ppEnabledExtensionNames = g_device_extensions.data();
    if (g_enable_validation_layers) {
      device_info.enabledLayerCount = static_cast<uint32_t>(
          g_validation_layers.size());
      device_info.ppEnabledLayerNames = g_validation_layers.data();
    }
    else {
      device_info.enabledLayerCount = 0;
    }
    auto res = m_phys_device.createDevice(&device_info, nullptr, &m_device);
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create logical device");
    }

    m_device.getQueue(indices.graphics_family.value(), 0, &m_graphics_queue);
    m_device.getQueue(indices.present_family.value(), 0, &m_present_queue);
  }

  bool isDeviceSuitable(const vk::PhysicalDevice& device) {
    if (!findQueueFamilies(device).allAvailable()) {
      return false;
    }
    if (!checkDeviceExtensionSupport(device)) {
      return false;
    }
    return true;
  }

  bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
    uint32_t n_extension;
    auto ret = device.enumerateDeviceExtensionProperties(nullptr, &n_extension, nullptr);
    if (ret != vk::Result::eSuccess) {
      throw std::runtime_error("");
    }
    std::vector<vk::ExtensionProperties> extensions(n_extension);
    ret = device.enumerateDeviceExtensionProperties(nullptr, &n_extension, extensions.data());
    if (ret != vk::Result::eSuccess) {
      throw std::runtime_error("");
    }
    std::set<std::string> required_extensions(g_device_extensions.begin(), g_device_extensions.end());
    for (const auto& extension : extensions) {
      required_extensions.erase(extension.extensionName);
    }
    return required_extensions.empty();
  }

  QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
    QueueFamilyIndices indices = {};
    uint32_t  n_queue_families = 0;
    device.getQueueFamilyProperties(&n_queue_families, nullptr);
    std::vector<vk::QueueFamilyProperties> queue_families(n_queue_families);
    device.getQueueFamilyProperties(&n_queue_families, queue_families.data());
    for (int i = 0; i < (int)queue_families.size(); ++i) {
      if (indices.allAvailable()) {
        break;
      }
      // queue for graphics commands
      if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphics_family = i;
      }
      // queue for present commands
      VkBool32 present_support = false;
      auto ret = device.getSurfaceSupportKHR(i, m_surface, &present_support);
      if (ret != vk::Result::eSuccess) {
        throw std::runtime_error("");
      }
      if (present_support) {
        indices.present_family = i;
      }
    }
    return indices;
  }

  bool checkValidationLayerSupport() {
    uint32_t n_layer;
    auto res = vk::enumerateInstanceLayerProperties(&n_layer, nullptr);
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("enumerateInstanceLayerProperties");
    }
    std::vector<vk::LayerProperties> layers(n_layer);
    res = vk::enumerateInstanceLayerProperties(&n_layer, layers.data());
    if (res != vk::Result::eSuccess) {
      throw std::runtime_error("enumerateInstanceLayerProperties");
    }

    for (const char* layer : g_validation_layers) {
      bool found = false;
      for (const auto& layer_props : layers) {
        if (strcmp(layer, layer_props.layerName) == 0) {
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "Missing validation layer " << layer << "\n";
        return false;
      }
    }
    std::cout << "All validation layers found.\n";
    return true;
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(m_window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    m_device.destroy(nullptr);
    m_instance.destroySurfaceKHR(m_surface, nullptr);
    m_instance.destroy(nullptr);
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  GLFWwindow* m_window;
  // vk::SurfaceKHR m_surface;
  VkSurfaceKHR m_surface;
  vk::Queue m_graphics_queue;
  vk::Queue m_present_queue;
  vk::Instance m_instance;
  vk::PhysicalDevice m_phys_device = VK_NULL_HANDLE;
  vk::Device m_device;
};

int main() {
  Application app;
  try {
    app.run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
