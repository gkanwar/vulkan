find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(BINARY ${PROJECT_NAME}.exe)

add_executable(${BINARY} "main.cpp")

target_compile_options(
  ${BINARY}
  PRIVATE -Wall -Wextra -Wpedantic -Werror
)

target_include_directories(
  ${BINARY}
  PUBLIC ${VULKAN_INCLUDE_DIRS}
)

target_link_libraries(
  ${BINARY}
  glfw
  ${Vulkan_LIBRARIES}
  shaders
)
