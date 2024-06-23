#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

// clip depth to [0,1] range
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <array>
#include <iostream>
#include <optional>
#include <set>
#include <vector>

const std::vector<const char*> g_validation_layers = {
  "VK_LAYER_KHRONOS_validation",
};
#ifndef NDEBUG
constexpr bool ENABLE_VALIDATION_LAYERS = true;
#else
constexpr bool ENABLE_VALIDATION_LAYERS = false;
#endif

constexpr uint64_t SECOND_NS = 1000000000;
constexpr uint64_t TIMEOUT = 10*SECOND_NS;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

extern const uint8_t _binary_shader_vert_spv_start[];
extern const uint8_t _binary_shader_vert_spv_end[];
extern const uint8_t _binary_shader_frag_spv_start[];
extern const uint8_t _binary_shader_frag_spv_end[];
const size_t vert_size = (size_t)_binary_shader_vert_spv_end - (size_t)_binary_shader_vert_spv_start;
const size_t frag_size = (size_t)_binary_shader_frag_spv_end - (size_t)_binary_shader_frag_spv_start;
// TODO: linker has issues with relocations for these
// extern const unsigned _binary_shader_vert_spv_size;
// extern const unsigned _binary_shader_frag_spv_size;

template<typename T>
vk::IndexType getIndexType();
template<>
vk::IndexType getIndexType<uint16_t>() {
  return vk::IndexType::eUint16;
}
template<>
vk::IndexType getIndexType<uint32_t>() {
  return vk::IndexType::eUint32;
}

template<typename T>
vk::Format getFormat();
template<>
vk::Format getFormat<glm::vec2>() {
  return vk::Format::eR32G32Sfloat;
}
template<>
vk::Format getFormat<glm::vec3>() {
  return vk::Format::eR32G32B32Sfloat;
}
template<>
vk::Format getFormat<glm::vec4>() {
  return vk::Format::eR32G32B32A32Sfloat;
}

const std::vector<const char*> g_device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

void check(vk::Result res, std::string msg) {
  if (res != vk::Result::eSuccess) {
    throw std::runtime_error(msg);
  }
}

void check(VkResult res, std::string msg) {
  if (res != VK_SUCCESS) {
    throw std::runtime_error(msg);
  }
}

template<typename T>
size_t sizeof_vec(const std::vector<T>& v) {
  return sizeof(T) * v.size();
}


// struct-of-arrays mesh
struct Mesh {
  std::vector<glm::vec3> xs;
  std::vector<glm::vec3> colors;
  std::vector<uint32_t> inds;

  // TODO: abstract this, coalesce device memory?
  std::optional<vk::Buffer> xs_buffer, xs_buffer_staging;
  std::optional<vk::Buffer> colors_buffer, colors_buffer_staging;
  std::optional<vk::Buffer> inds_buffer, inds_buffer_staging;
  std::optional<vk::DeviceMemory> xs_mem, xs_mem_staging;
  std::optional<vk::DeviceMemory> colors_mem, colors_mem_staging;
  std::optional<vk::DeviceMemory> inds_mem, inds_mem_staging;

  static std::array<vk::VertexInputBindingDescription, 2>
  getBindingDescriptions() {
    vk::VertexInputBindingDescription desc_x = {};
    desc_x.binding = 0;
    desc_x.stride = sizeof(glm::vec3);
    desc_x.inputRate = vk::VertexInputRate::eVertex;
    vk::VertexInputBindingDescription desc_c = {};
    desc_c.binding = 1;
    desc_c.stride = sizeof(glm::vec3);
    desc_c.inputRate = vk::VertexInputRate::eVertex;
    return {desc_x, desc_c};
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    vk::VertexInputAttributeDescription desc_x = {};
    desc_x.binding = 0;
    desc_x.location = 0;
    desc_x.format = getFormat<glm::vec3>(); // vk::Format::eR32G32B32Sfloat;
    desc_x.offset = 0;
    vk::VertexInputAttributeDescription desc_c = {};
    desc_c.binding = 1;
    desc_c.location = 1;
    desc_c.format = getFormat<glm::vec3>(); // vk::Format::eR32G32B32Sfloat;
    desc_c.offset = 0;
    return {desc_x, desc_c};
  }
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool allAvailable() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR caps;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> modes;
  bool isAcceptable() {
    return !formats.empty() && !modes.empty();
  }
};

class Application {
 public:
  void run() {
    initMeshes();
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  void initMeshes() {
    // triangles
    m_meshes.push_back({});
    m_meshes.back().xs = {
      glm::vec3(0.5, -0.5, 0.0),
      glm::vec3(-0.5, 0.5, 0.0),
      glm::vec3(0.5, 0.5, 0.0),
      glm::vec3(-0.5, -0.5, 0.0),
    };
    m_meshes.back().colors = {
      glm::vec3(1.0, 1.0, 1.0),
      glm::vec3(0.0, 1.0, 0.0),
      glm::vec3(0.0, 0.0, 1.0),
      glm::vec3(1.0, 0.0, 0.0),
    };
    m_meshes.back().inds = {
      0, 1, 2,
      1, 0, 3,
    };
  }

  void initWindow() {
    glfwInit();
    // no OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(800, 600, "Hello triangle", nullptr, nullptr);
    // resize handler
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResized);
  }

  static void framebufferResized(
      GLFWwindow* window, [[maybe_unused]] int width, [[maybe_unused]] int height) {
    auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_fb_resized = true;
  }

  void initVulkan() {
    createVkInstance();
    createVkSurface();
    selectVkPhysicalDevice();
    createVkLogicalDevice();
    createVkSwapchain();
    createVkImageViews();
    createVkRenderPass();
    createVkGraphicsPipeline();
    createVkFramebuffers();
    createVkCommandPool();
    // TODO: allow meshes to be added/removed dynamically
    createVkVertexBuffers(m_meshes);
    createVkCommandBuffers();
    createVkSyncObjects();
  }

  void recreateVkSwapchain() {
    // pause until we have a non-trivial draw surface (e.g. wait until not minimized)
    int width = 0, height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(m_window, &width, &height);
      glfwWaitEvents();
    }

    m_device.waitIdle();
    cleanupVkSwapchain();
    createVkSwapchain();
    createVkImageViews();
    createVkFramebuffers();
  }

  void createVkInstance() {
    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
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
    if (ENABLE_VALIDATION_LAYERS) {
      inst_info.enabledLayerCount = g_validation_layers.size();
      inst_info.ppEnabledLayerNames = g_validation_layers.data();
    }

    auto res = vk::createInstance(&inst_info, nullptr, &m_instance);
    check(res, "createInstance");
  }

  void createVkSurface() {
    auto res = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    check(res, "failed to create window surface");
  }

  void selectVkPhysicalDevice() {
    uint32_t n_device = 0;
    auto res = m_instance.enumeratePhysicalDevices(&n_device, nullptr);
    check(res, "");
    if (n_device == 0) {
      throw std::runtime_error("no supported Vulkan devices available");
    }
    std::vector<vk::PhysicalDevice> devices(n_device);
    res = m_instance.enumeratePhysicalDevices(&n_device, devices.data());
    check(res, "");
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

  void createVkLogicalDevice() {
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
    if (ENABLE_VALIDATION_LAYERS) {
      device_info.enabledLayerCount = static_cast<uint32_t>(
          g_validation_layers.size());
      device_info.ppEnabledLayerNames = g_validation_layers.data();
      std::cout << "Creating device with validation layers\n";
    }
    else {
      device_info.enabledLayerCount = 0;
      std::cout << "Creating device with no validation\n";
    }
    auto res = m_phys_device.createDevice(&device_info, nullptr, &m_device);
    check(res, "failed to create logical device");

    m_device.getQueue(indices.graphics_family.value(), 0, &m_graphics_queue);
    m_device.getQueue(indices.present_family.value(), 0, &m_present_queue);
  }

  void createVkSwapchain() {
    SwapChainSupportDetails swap_chain_support = querySwapChainSupportKHR(m_phys_device);
    m_format = selectSwapSurfaceFormatKHR(swap_chain_support.formats);
    m_present_mode = selectSwapPresentModeKHR(swap_chain_support.modes);
    m_extent = selectSwapExtentKHR(swap_chain_support.caps);
    uint32_t n_image = swap_chain_support.caps.minImageCount + 1;
    n_image = std::min(n_image, swap_chain_support.caps.maxImageCount);

    vk::SwapchainCreateInfoKHR info = {};
    info.sType =  vk::StructureType::eSwapchainCreateInfoKHR;
    info.surface = m_surface;
    info.minImageCount = n_image;
    info.imageFormat = m_format.format;
    info.imageColorSpace = m_format.colorSpace;
    info.imageExtent = m_extent;
    info.imageArrayLayers = 1;
    // color attachment: direct render into image
    // vs. transfer destination: copy from intermediate
    info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = findQueueFamilies(m_phys_device);
    uint32_t queue_family_indices[] = {
      indices.graphics_family.value(),
      indices.present_family.value(),
    };
    // for convenience, use shared access mode when queues are distinct
    // vs. more performant explicit handoffs of exclusive access
    if (indices.graphics_family != indices.present_family) {
      info.imageSharingMode = vk::SharingMode::eConcurrent;
      info.queueFamilyIndexCount = 2;
      info.pQueueFamilyIndices = queue_family_indices;
    }
    else {
      info.imageSharingMode = vk::SharingMode::eExclusive;
    }

    info.preTransform = swap_chain_support.caps.currentTransform;
    info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    info.presentMode = m_present_mode;
    info.clipped = vk::True;
    // in the future may need this to recreate swapchain during execution
    info.oldSwapchain = VK_NULL_HANDLE;

    auto res = m_device.createSwapchainKHR(&info, nullptr, &m_swapchain);
    check(res, "failed to create swap chain");

    res = m_device.getSwapchainImagesKHR(m_swapchain, &n_image, nullptr);
    check(res, "getSwapchainImagesKHR");
    m_swap_images.resize(n_image);
    res = m_device.getSwapchainImagesKHR(m_swapchain, &n_image, m_swap_images.data());
    check(res, "getSwapchainImagesKHR");
  }

  void createVkImageViews() {
    m_swap_image_views.resize(m_swap_images.size());
    for (size_t i = 0; i < m_swap_images.size(); ++i) {
      vk::ImageViewCreateInfo info = {};
      info.sType = vk::StructureType::eImageViewCreateInfo;
      info.image = m_swap_images[i];
      info.viewType = vk::ImageViewType::e2D;
      info.format = m_format.format;
      info.components.r = vk::ComponentSwizzle::eIdentity;
      info.components.g = vk::ComponentSwizzle::eIdentity;
      info.components.b = vk::ComponentSwizzle::eIdentity;
      info.components.a = vk::ComponentSwizzle::eIdentity;
      info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      info.subresourceRange.baseMipLevel = 0;
      info.subresourceRange.levelCount = 1;
      info.subresourceRange.baseArrayLayer = 0;
      info.subresourceRange.layerCount = 1;
      auto res = m_device.createImageView(&info, nullptr, &m_swap_image_views[i]);
      check(res, "failed to create image view");
    }
  }

  void createVkRenderPass() {
    vk::AttachmentDescription color_attach;
    color_attach.format = m_format.format;
    color_attach.samples = vk::SampleCountFlagBits::e1;
    color_attach.loadOp = vk::AttachmentLoadOp::eClear;
    color_attach.storeOp = vk::AttachmentStoreOp::eStore;
    color_attach.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    color_attach.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    color_attach.initialLayout = vk::ImageLayout::eUndefined;
    color_attach.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attach_ref;
    color_attach_ref.attachment = 0;
    color_attach_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;
    vk::SubpassDescription subpass;
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attach_ref;

    // wait on color attachment output stage from before this render pass
    vk::SubpassDependency dep = {};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.srcAccessMask = {};
    dep.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo info = {};
    info.sType = vk::StructureType::eRenderPassCreateInfo;
    info.attachmentCount = 1;
    info.pAttachments = &color_attach;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dep;

    auto res = m_device.createRenderPass(&info, nullptr, &m_render_pass);
    check(res, "createRenderPass");
  }

  void createVkGraphicsPipeline() {
    std::cout << "Built with vertex shader (" << vert_size << ")\n";
    std::cout << "Built with frag shader (" << frag_size << ")\n";
    std::vector<char> vert_code(_binary_shader_vert_spv_start, _binary_shader_vert_spv_end);
    std::vector<char> frag_code(_binary_shader_frag_spv_start, _binary_shader_frag_spv_end);
    vk::ShaderModule vert_mod = createShaderModule(vert_code);
    vk::ShaderModule frag_mod = createShaderModule(frag_code);

    // stage: vertex shader
    vk::PipelineShaderStageCreateInfo info_v = {};
    info_v.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
    info_v.stage = vk::ShaderStageFlagBits::eVertex;
    info_v.module = vert_mod;
    info_v.pName = "main";
    // could implement compile-time specialization here
    info_v.pSpecializationInfo = nullptr;

    // stage: frag shader
    vk::PipelineShaderStageCreateInfo info_f = {};
    info_f.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
    info_f.stage = vk::ShaderStageFlagBits::eFragment;
    info_f.module = frag_mod;
    info_f.pName = "main";
    // could implement compile-time specialization here
    info_f.pSpecializationInfo = nullptr;

    vk::PipelineShaderStageCreateInfo shader_stages[] = {info_v, info_f};



    // dynamic state
    std::vector<vk::DynamicState> dynamic_states = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
    };
    vk::PipelineDynamicStateCreateInfo info_dyn = {};
    info_dyn.sType = vk::StructureType::ePipelineDynamicStateCreateInfo;
    info_dyn.dynamicStateCount = dynamic_states.size();
    info_dyn.pDynamicStates = dynamic_states.data();

    // stage: vertex input
    vk::PipelineVertexInputStateCreateInfo info_vin = {};
    info_vin.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
    auto bindings = Mesh::getBindingDescriptions();
    auto attributes = Mesh::getAttributeDescriptions();
    info_vin.vertexBindingDescriptionCount = 2;
    info_vin.pVertexBindingDescriptions = bindings.data();
    info_vin.vertexAttributeDescriptionCount = 2;
    info_vin.pVertexAttributeDescriptions = attributes.data();

    // stage: input assembly
    vk::PipelineInputAssemblyStateCreateInfo info_asm = {};
    info_asm.sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo;
    info_asm.topology = vk::PrimitiveTopology::eTriangleList;
    info_asm.primitiveRestartEnable = vk::False;

    // stage: viewport state
    vk::PipelineViewportStateCreateInfo info_vp = {};
    info_vp.sType = vk::StructureType::ePipelineViewportStateCreateInfo;
    info_vp.viewportCount = 1;
    info_vp.scissorCount = 1;

    // stage: rasterization
    vk::PipelineRasterizationStateCreateInfo info_rast = {};
    info_rast.sType = vk::StructureType::ePipelineRasterizationStateCreateInfo;
    info_rast.depthClampEnable = vk::False;
    info_rast.rasterizerDiscardEnable = vk::False;
    info_rast.polygonMode = vk::PolygonMode::eFill;
    info_rast.lineWidth = 1.0f;
    info_rast.cullMode = vk::CullModeFlagBits::eBack;
    info_rast.frontFace = vk::FrontFace::eCounterClockwise;
    info_rast.depthBiasEnable = vk::False;

    // stage: multisampling
    vk::PipelineMultisampleStateCreateInfo info_ms = {};
    info_ms.sType = vk::StructureType::ePipelineMultisampleStateCreateInfo;
    info_ms.sampleShadingEnable = vk::False;
    info_ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

    // stage: depth/stencil testing
    vk::PipelineDepthStencilStateCreateInfo info_ds = {};
    info_ds.sType = vk::StructureType::ePipelineDepthStencilStateCreateInfo;

    // stage: color blending
    vk::PipelineColorBlendAttachmentState cb_attachment = {};
    cb_attachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    cb_attachment.blendEnable = vk::True;
    cb_attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    cb_attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    cb_attachment.colorBlendOp = vk::BlendOp::eAdd;
    cb_attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    cb_attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    cb_attachment.alphaBlendOp = vk::BlendOp::eAdd;
    vk::PipelineColorBlendStateCreateInfo info_cb = {};
    info_cb.sType = vk::StructureType::ePipelineColorBlendStateCreateInfo;
    info_cb.logicOpEnable = vk::False;
    info_cb.attachmentCount = 1;
    info_cb.pAttachments = &cb_attachment;

    // pipeline layout
    vk::PipelineLayoutCreateInfo info_pp = {};
    info_pp.sType = vk::StructureType::ePipelineLayoutCreateInfo;
    // any push constants or uniforms go here
    auto res = m_device.createPipelineLayout(&info_pp, nullptr, &m_pipeline_layout);
    check(res, "createPipelineLayout");

    vk::GraphicsPipelineCreateInfo info = {};
    info.sType = vk::StructureType::eGraphicsPipelineCreateInfo;
    info.stageCount = 2;
    info.pStages = shader_stages;
    info.pVertexInputState = &info_vin;
    info.pInputAssemblyState = &info_asm;
    info.pViewportState = &info_vp;
    info.pRasterizationState = &info_rast;
    info.pMultisampleState = &info_ms;
    info.pDepthStencilState = &info_ds;
    info.pColorBlendState = &info_cb;
    info.pDynamicState = &info_dyn;
    info.layout = m_pipeline_layout;
    info.renderPass = m_render_pass;
    info.subpass = 0;

    info.basePipelineHandle = VK_NULL_HANDLE;
    info.basePipelineIndex = -1;

    res = m_device.createGraphicsPipelines(VK_NULL_HANDLE, 1, &info, nullptr, &m_pipeline);
    check(res, "createGraphicsPipelines");

    m_device.destroy(vert_mod, nullptr);
    m_device.destroy(frag_mod, nullptr);
  }

  void createVkFramebuffers() {
    m_swap_fbs.resize(m_swap_image_views.size());
    for (size_t i = 0; i < m_swap_image_views.size(); ++i) {
      vk::FramebufferCreateInfo info = {};
      info.sType = vk::StructureType::eFramebufferCreateInfo;
      info.renderPass = m_render_pass;
      info.attachmentCount = 1;
      info.pAttachments = &m_swap_image_views[i];
      info.width = m_extent.width;
      info.height = m_extent.height;
      info.layers = 1;

      auto res = m_device.createFramebuffer(&info, nullptr, &m_swap_fbs[i]);
      check(res, "createFramebuffer");
    }
  }

  void createVkCommandPool() {
    QueueFamilyIndices queue_family_indices = findQueueFamilies(m_phys_device);
    vk::CommandPoolCreateInfo info = {};
    info.sType = vk::StructureType::eCommandPoolCreateInfo;
    info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    info.queueFamilyIndex = queue_family_indices.graphics_family.value();
    auto res = m_device.createCommandPool(&info, nullptr, &m_cmd_pool);
    check(res, "createCommandPool");
  }

  void createVkBuffer(
      vk::DeviceSize size, vk::BufferUsageFlags usage_flags,
      vk::MemoryPropertyFlags mem_flags,
      vk::Buffer& buffer, vk::DeviceMemory& mem) {
    vk::BufferCreateInfo info_buf = {};
    info_buf.sType = vk::StructureType::eBufferCreateInfo;
    info_buf.size = size;
    info_buf.usage = usage_flags;
    // exclusive to the graphics queue
    info_buf.sharingMode = vk::SharingMode::eExclusive;
    auto res = m_device.createBuffer(&info_buf, nullptr, &buffer);
    check(res, "createBuffer");

    vk::MemoryRequirements mem_reqs;
    m_device.getBufferMemoryRequirements(buffer, &mem_reqs);
    vk::MemoryAllocateInfo info_mem = {};
    info_mem.sType = vk::StructureType::eMemoryAllocateInfo;
    info_mem.allocationSize = mem_reqs.size;
    info_mem.memoryTypeIndex = findMemoryType(mem_reqs.memoryTypeBits, mem_flags);
    res = m_device.allocateMemory(&info_mem, nullptr, &mem);
    check(res, "allocateMemory");

    m_device.bindBufferMemory(buffer, mem, 0);
  }

  void createVkVertexBuffers(std::vector<Mesh>& meshes) {
    std::vector<vk::Fence> xfer_fences;
    std::vector<vk::CommandBuffer> xfer_cmd_bufs;
    
    for (auto& mesh : meshes) {
      auto usage_staging = vk::BufferUsageFlagBits::eTransferSrc;
      auto usage_verts_dst = vk::BufferUsageFlagBits::eVertexBuffer
          | vk::BufferUsageFlagBits::eTransferDst;
      auto usage_inds_dst = vk::BufferUsageFlagBits::eIndexBuffer
          | vk::BufferUsageFlagBits::eTransferDst;
      auto mem_flags_staging = vk::MemoryPropertyFlagBits::eHostVisible
          | vk::MemoryPropertyFlagBits::eHostCoherent;
      auto mem_flags_dst = vk::MemoryPropertyFlagBits::eDeviceLocal;
      // position buffer
      {
        vk::Buffer buffer_staging, buffer_dst;
        vk::DeviceMemory mem_staging, mem_dst;
        uint32_t size = sizeof_vec(mesh.xs);
        createVkBuffer(
            size, usage_staging, mem_flags_staging,
            buffer_staging, mem_staging);
        mesh.xs_buffer_staging = buffer_staging;
        mesh.xs_mem_staging = mem_staging;

        void* xs_mmap;
        auto res = m_device.mapMemory(mem_staging, 0, size, {}, &xs_mmap);
        check(res, "failed to map GPU buffer");
        memcpy(xs_mmap, mesh.xs.data(), size);
        // no flush required because we requested coherent memory alloc
        m_device.unmapMemory(mem_staging);

        createVkBuffer(size, usage_verts_dst, mem_flags_dst, buffer_dst, mem_dst);
        mesh.xs_buffer = buffer_dst;
        mesh.xs_mem = mem_dst;
        copyBuffer(buffer_staging, buffer_dst, size, xfer_cmd_bufs, xfer_fences);
      }
      // non-position buffer (colors, normals, etc.)
      {
        vk::Buffer buffer_staging, buffer_dst;
        vk::DeviceMemory mem_staging, mem_dst;
        uint32_t size = sizeof_vec(mesh.colors);
        createVkBuffer(
            size, usage_staging, mem_flags_staging,
            buffer_staging, mem_staging);
        mesh.colors_buffer_staging = buffer_staging;
        mesh.colors_mem_staging = mem_staging;

        void* colors_mmap;
        auto res = m_device.mapMemory(mem_staging, 0, size, {}, &colors_mmap);
        check(res, "failed to map GPU buffer");
        memcpy(colors_mmap, mesh.colors.data(), size);
        // no flush required because we requested coherent memory alloc
        m_device.unmapMemory(mem_staging);

        createVkBuffer(size, usage_verts_dst, mem_flags_dst, buffer_dst, mem_dst);
        mesh.colors_buffer = buffer_dst;
        mesh.colors_mem = mem_dst;
        copyBuffer(buffer_staging, buffer_dst, size, xfer_cmd_bufs, xfer_fences);
      }
      // indices buffer
      {
        vk::Buffer buffer_staging, buffer_dst;
        vk::DeviceMemory mem_staging, mem_dst;
        uint32_t size = sizeof_vec(mesh.inds);
        createVkBuffer(
            size, usage_staging, mem_flags_staging,
            buffer_staging, mem_staging);
        mesh.inds_buffer_staging = buffer_staging;
        mesh.inds_mem_staging = mem_staging;

        void* inds_mmap;
        auto res = m_device.mapMemory(mem_staging, 0, size, {}, &inds_mmap);
        check(res, "failed to map GPU buffer");
        memcpy(inds_mmap, mesh.inds.data(), size);
        // no flush required because we requested coherent memory alloc
        m_device.unmapMemory(mem_staging);

        createVkBuffer(size, usage_inds_dst, mem_flags_dst, buffer_dst, mem_dst);
        mesh.inds_buffer = buffer_dst;
        mesh.inds_mem = mem_dst;
        copyBuffer(buffer_staging, buffer_dst, size, xfer_cmd_bufs, xfer_fences);
      }
    }

    auto res = m_device.waitForFences(xfer_fences.size(), xfer_fences.data(), vk::True, TIMEOUT);
    check(res, "waitForFences");

    for (auto& cmd_buf : xfer_cmd_bufs) {
      m_device.freeCommandBuffers(m_cmd_pool, 1, &cmd_buf);
    }
    for (auto& fence : xfer_fences) {
      m_device.destroyFence(fence, nullptr);
    }

    for (auto& mesh : meshes) {
      assert(mesh.xs_buffer_staging && mesh.xs_mem_staging);
      assert(mesh.colors_buffer_staging && mesh.colors_mem_staging);
      assert(mesh.inds_buffer_staging && mesh.inds_mem_staging);
      m_device.destroyBuffer(mesh.xs_buffer_staging.value(), nullptr);
      m_device.freeMemory(mesh.xs_mem_staging.value(), nullptr);
      m_device.destroyBuffer(mesh.colors_buffer_staging.value(), nullptr);
      m_device.freeMemory(mesh.colors_mem_staging.value(), nullptr);
      m_device.destroyBuffer(mesh.inds_buffer_staging.value(), nullptr);
      m_device.freeMemory(mesh.inds_mem_staging.value(), nullptr);
      mesh.xs_buffer_staging.reset();
      mesh.xs_mem_staging.reset();
      mesh.colors_buffer_staging.reset();
      mesh.colors_mem_staging.reset();
      mesh.inds_buffer_staging.reset();
      mesh.inds_mem_staging.reset();
    }
  }

  void copyBuffer(
      vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
      std::vector<vk::CommandBuffer>& cmd_bufs, std::vector<vk::Fence>& fences) {
    vk::CommandBufferAllocateInfo info = {};
    info.sType = vk::StructureType::eCommandBufferAllocateInfo;
    info.level = vk::CommandBufferLevel::ePrimary;
    info.commandPool = m_cmd_pool;
    info.commandBufferCount = 1;

    vk::CommandBuffer cmd_buf;
    auto res = m_device.allocateCommandBuffers(&info, &cmd_buf);
    check(res, "allocateCommandBuffers");

    vk::CommandBufferBeginInfo info_begin = {};
    info_begin.sType = vk::StructureType::eCommandBufferBeginInfo;
    info_begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    // build copy command buf
    res = cmd_buf.begin(&info_begin);
    check(res, "failed to begin command buffer");
    vk::BufferCopy info_copy = {};
    info_copy.srcOffset = 0;
    info_copy.dstOffset = 0;
    info_copy.size = size;
    cmd_buf.copyBuffer(src, dst, 1, &info_copy);
    cmd_buf.end();

    vk::SubmitInfo info_submit = {};
    info_submit.sType = vk::StructureType::eSubmitInfo;
    info_submit.commandBufferCount = 1;
    info_submit.pCommandBuffers = &cmd_buf;

    vk::FenceCreateInfo info_fence = {};
    info_fence.sType = vk::StructureType::eFenceCreateInfo;
    vk::Fence xfer_fence;
    res = m_device.createFence(&info_fence, nullptr, &xfer_fence);
    check(res, "createFence");

    res = m_graphics_queue.submit(1, &info_submit, xfer_fence);
    check(res, "failed to submit command buffer");

    // save for later cleanup
    cmd_bufs.push_back(cmd_buf);
    fences.push_back(xfer_fence);
  }

  void createVkCommandBuffers() {
    vk::CommandBufferAllocateInfo info = {};
    info.sType = vk::StructureType::eCommandBufferAllocateInfo;
    info.commandPool = m_cmd_pool;
    info.level = vk::CommandBufferLevel::ePrimary;
    info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    m_cmd_buf.resize(MAX_FRAMES_IN_FLIGHT);
    auto res = m_device.allocateCommandBuffers(&info, m_cmd_buf.data());
    check(res, "allocateCommandBuffers");
  }

  void createVkSyncObjects() {
    vk::SemaphoreCreateInfo info_sem = {};
    info_sem.sType = vk::StructureType::eSemaphoreCreateInfo;
    vk::FenceCreateInfo info_fence = {};
    info_fence.sType = vk::StructureType::eFenceCreateInfo;
    // fence starts signaled
    info_fence.flags = vk::FenceCreateFlagBits::eSignaled;
    m_sem_image_avail.resize(MAX_FRAMES_IN_FLIGHT);
    m_sem_render_done.resize(MAX_FRAMES_IN_FLIGHT);
    m_fence_in_flight.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      auto res = m_device.createSemaphore(&info_sem, nullptr, &m_sem_image_avail[i]);
      check(res, "createSemaphore");
      res = m_device.createSemaphore(&info_sem, nullptr, &m_sem_render_done[i]);
      check(res, "createSemaphore");
      res = m_device.createFence(&info_fence, nullptr, &m_fence_in_flight[i]);
      check(res, "createFence");
    }
  }

  void recordCommandBuffer(vk::CommandBuffer& cmd_buf, uint32_t img_index) {
    // begin cmd buffer
    {
      vk::CommandBufferBeginInfo info = {};
      info.sType = vk::StructureType::eCommandBufferBeginInfo;
      info.flags = {};
      info.pInheritanceInfo = nullptr;
      auto res = cmd_buf.begin(&info);
      check(res, "failed to start recording commands");
    }
    // begin render pass
    {
      vk::RenderPassBeginInfo info = {};
      info.sType = vk::StructureType::eRenderPassBeginInfo;
      info.renderPass = m_render_pass;
      info.framebuffer = m_swap_fbs[img_index];
      info.renderArea.offset = vk::Offset2D{0, 0};
      info.renderArea.extent = m_extent;
      vk::ClearValue clear_color = {{0.1f, 0.1f, 0.1f, 1.0f}};
      info.clearValueCount = 1;
      info.pClearValues = &clear_color;
      cmd_buf.beginRenderPass(&info, vk::SubpassContents::eInline);
    }

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

    vk::Viewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)m_extent.width;
    viewport.height = (float)m_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    cmd_buf.setViewport(0, 1, &viewport);
    vk::Rect2D scissor = {};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = m_extent;
    cmd_buf.setScissor(0, 1, &scissor);

    // TODO: "bindless" rendering with one large buffer shared across all meshes
    for (const auto& mesh : m_meshes) {
      vk::Buffer vert_buffers[] = {mesh.xs_buffer.value(), mesh.colors_buffer.value()};
      vk::DeviceSize offsets[] = {0, 0};

      const uint32_t off = 0;
      const uint32_t n_bindings = 2;
      cmd_buf.bindVertexBuffers(off, n_bindings, vert_buffers, offsets);

      auto idx_type = getIndexType<decltype(mesh.inds)::value_type>();
      cmd_buf.bindIndexBuffer(mesh.inds_buffer.value(), 0, idx_type);

      const size_t n_inst = 1;
      const uint32_t n_idx = mesh.inds.size();
      const size_t inst_off = 0;
      const size_t idx_off = 0;
      const size_t idx_shift = 0;
      cmd_buf.drawIndexed(n_idx, n_inst, idx_off, idx_shift, inst_off);
    }

    cmd_buf.endRenderPass();
    cmd_buf.end();
  }

  vk::ShaderModule createShaderModule(const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo info = {};
    info.sType = vk::StructureType::eShaderModuleCreateInfo;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    vk::ShaderModule mod;
    auto res = m_device.createShaderModule(&info, nullptr, &mod);
    check(res, "createShaderModule");
    return mod;
  }

  bool isDeviceSuitable(const vk::PhysicalDevice& device) {
    if (!findQueueFamilies(device).allAvailable()) {
      return false;
    }
    if (!checkDeviceExtensionSupport(device)) {
      return false;
    }
    if (!querySwapChainSupportKHR(device).isAcceptable()) {
      return false;
    }
    return true;
  }

  bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
    uint32_t n_extension;
    auto res = device.enumerateDeviceExtensionProperties(nullptr, &n_extension, nullptr);
    check(res, "");
    std::vector<vk::ExtensionProperties> extensions(n_extension);
    res = device.enumerateDeviceExtensionProperties(nullptr, &n_extension, extensions.data());
    check(res, "");
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

  uint32_t findMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags flags) {
    vk::PhysicalDeviceMemoryProperties props;
    m_phys_device.getMemoryProperties(&props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      // restrict the allowable types
      if (!(type_filter & (1 << i))) {
        continue;
      }
      // all required properties are available
      if ((props.memoryTypes[i].propertyFlags & flags) != flags) {
        continue;
      }
      return i;
    }

    throw std::runtime_error("failed to find suitable memory");
  }

  SwapChainSupportDetails querySwapChainSupportKHR(const vk::PhysicalDevice& device) {
    SwapChainSupportDetails details;
    // capabilities
    auto res = device.getSurfaceCapabilitiesKHR(m_surface, &details.caps);
    check(res, "getSurfaceCapabilitiesKHR");
    // formats
    uint32_t n_format;
    res = device.getSurfaceFormatsKHR(m_surface, &n_format, nullptr);
    check(res, "getSurfaceFormatsKHR");
    details.formats.resize(n_format);
    res = device.getSurfaceFormatsKHR(m_surface, &n_format, details.formats.data());
    check(res, "getSurfaceFormatsKHR");
    // present modes
    uint32_t n_modes;
    res = device.getSurfacePresentModesKHR(m_surface, &n_modes, nullptr);
    check(res, "getSurfacePresentModesKHR");
    details.modes.resize(n_modes);
    res = device.getSurfacePresentModesKHR(m_surface, &n_modes, details.modes.data());
    check(res, "getSurfacePresentModesKHR");
    return details;
  }

  vk::SurfaceFormatKHR selectSwapSurfaceFormatKHR(const std::vector<vk::SurfaceFormatKHR>& formats) {
    // try for BGRA8888 sRGB, otherwise select the first format
    for (const auto& format : formats) {
      if (format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return format;
      }
    }
    return formats[0];
  }

  vk::PresentModeKHR selectSwapPresentModeKHR(const std::vector<vk::PresentModeKHR>& modes) {
    // FORNOW: prefer immediate, as this is the only sensible mode on X11 + nvidia
    for (const auto& mode : modes) {
      if (mode == vk::PresentModeKHR::eImmediate) {
        return mode;
      }
    }
    return modes[0];
  }

  vk::Extent2D selectSwapExtentKHR(const vk::SurfaceCapabilitiesKHR& caps) {
    // extent set by Vulkan itself
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return caps.currentExtent;
    }
    // manually set extent to match window size in pixels
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    vk::Extent2D extent = {
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height)
    };
    uint32_t min_width = caps.minImageExtent.width;
    uint32_t min_height = caps.minImageExtent.height;
    uint32_t max_width = caps.maxImageExtent.width;
    uint32_t max_height = caps.maxImageExtent.height;
    extent.width = std::clamp(extent.width, min_width, max_width);
    extent.height = std::clamp(extent.height, min_height, max_height);
    return extent;
  }

  bool checkValidationLayerSupport() {
    uint32_t n_layer;
    auto res = vk::enumerateInstanceLayerProperties(&n_layer, nullptr);
    check(res, "enumerateInstanceLayerProperties");
    std::vector<vk::LayerProperties> layers(n_layer);
    res = vk::enumerateInstanceLayerProperties(&n_layer, layers.data());
    check(res, "enumerateInstanceLayerProperties");

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
      drawFrame();
    }
    m_device.waitIdle();
  }

  void drawFrame() {
    // sync
    auto res = m_device.waitForFences(1, &m_fence_in_flight[m_frame], vk::True, TIMEOUT);
    check(res, "waitForFences");

    // get swap chain index, record command buf
    uint32_t img_index;
    constexpr auto no_fence = VK_NULL_HANDLE;
    res = m_device.acquireNextImageKHR(
        m_swapchain, TIMEOUT, m_sem_image_avail[m_frame], no_fence, &img_index);
    if (res == vk::Result::eErrorOutOfDateKHR) {
      recreateVkSwapchain();
      return;
    }
    check(res, "acquireNextImageKHR");
    constexpr vk::CommandBufferResetFlags flags = {};
    m_cmd_buf[m_frame].reset(flags);
    recordCommandBuffer(m_cmd_buf[m_frame], img_index);

    // submit command buf
    vk::SubmitInfo info = {};
    info.sType = vk::StructureType::eSubmitInfo;

    vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &m_sem_image_avail[m_frame];
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &m_cmd_buf[m_frame];
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_sem_render_done[m_frame];

    res = m_device.resetFences(1, &m_fence_in_flight[m_frame]);
    check(res, "resetFences");
    res = m_graphics_queue.submit(1, &info, m_fence_in_flight[m_frame]);
    check(res, "failed to submit draw command buffer");

    // present frame
    vk::PresentInfoKHR info_present = {};
    info_present.sType = vk::StructureType::ePresentInfoKHR;
    info_present.waitSemaphoreCount = 1;
    info_present.pWaitSemaphores = &m_sem_render_done[m_frame];
    info_present.swapchainCount = 1;
    info_present.pSwapchains = &m_swapchain;
    info_present.pImageIndices = &img_index;

    res = m_present_queue.presentKHR(&info_present);
    if (res == vk::Result::eErrorOutOfDateKHR ||
        res == vk::Result::eSuboptimalKHR ||
        m_fb_resized) {
      m_fb_resized = false;
      recreateVkSwapchain();
    }
    else {
      check(res, "failed to present frame");
    }

    // advance frame
    m_frame = (m_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanupVkSwapchain() {
    for (auto fb : m_swap_fbs) {
      m_device.destroyFramebuffer(fb, nullptr);
    }
    for (auto image_view : m_swap_image_views) {
      m_device.destroyImageView(image_view, nullptr);
    }
    m_device.destroySwapchainKHR(m_swapchain, nullptr);
  }

  void cleanupVkVertexBuffers(std::vector<Mesh>& meshes) {
    for (auto& mesh : meshes) {
      if (mesh.xs_buffer) {
        m_device.destroyBuffer(mesh.xs_buffer.value(), nullptr);
        m_device.freeMemory(mesh.xs_mem.value(), nullptr);
        mesh.xs_buffer.reset();
        mesh.xs_mem.reset();
      }
      if (mesh.colors_buffer) {
        m_device.destroyBuffer(mesh.colors_buffer.value(), nullptr);
        m_device.freeMemory(mesh.colors_mem.value(), nullptr);
        mesh.colors_buffer.reset();
        mesh.colors_mem.reset();
      }
      if (mesh.inds_buffer) {
        m_device.destroyBuffer(mesh.inds_buffer.value(), nullptr);
        m_device.freeMemory(mesh.inds_mem.value(), nullptr);
        mesh.inds_buffer.reset();
        mesh.inds_mem.reset();
      }
    }
  }

  void cleanup() {
    cleanupVkSwapchain();
    cleanupVkVertexBuffers(m_meshes);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      m_device.destroySemaphore(m_sem_image_avail[i], nullptr);
      m_device.destroySemaphore(m_sem_render_done[i], nullptr);
      m_device.destroyFence(m_fence_in_flight[i], nullptr);
    }
    m_device.destroyCommandPool(m_cmd_pool, nullptr);
    m_device.destroyPipeline(m_pipeline, nullptr);
    m_device.destroyPipelineLayout(m_pipeline_layout, nullptr);
    m_device.destroyRenderPass(m_render_pass, nullptr);
    m_device.destroy(nullptr);
    m_instance.destroySurfaceKHR(m_surface, nullptr);
    m_instance.destroy(nullptr);
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  // glfw stuff
  GLFWwindow* m_window;
  // vulkan stuff
  vk::Queue m_graphics_queue;
  vk::Queue m_present_queue;
  vk::Instance m_instance;
  vk::PhysicalDevice m_phys_device = VK_NULL_HANDLE;
  vk::Device m_device;
  // swapchain
  std::vector<vk::Image> m_swap_images;
  std::vector<vk::ImageView> m_swap_image_views;
  std::vector<vk::Framebuffer> m_swap_fbs;
  vk::SwapchainKHR m_swapchain;
  // surface properties
  VkSurfaceKHR m_surface;
  vk::SurfaceFormatKHR m_format;
  vk::PresentModeKHR m_present_mode;
  vk::Extent2D m_extent;
  // pipeline
  vk::PipelineLayout m_pipeline_layout;
  vk::RenderPass m_render_pass;
  vk::Pipeline m_pipeline;
  // drawing
  vk::CommandPool m_cmd_pool;
  std::vector<vk::CommandBuffer> m_cmd_buf;
  uint32_t m_frame = 0;
  bool m_fb_resized = false;
  // sync
  std::vector<vk::Semaphore> m_sem_image_avail;
  std::vector<vk::Semaphore> m_sem_render_done;
  std::vector<vk::Fence> m_fence_in_flight;
  // data
  std::vector<Mesh> m_meshes;
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
