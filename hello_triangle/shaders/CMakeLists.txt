# https://stackoverflow.com/a/68457439/234256

find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_exe NAMES glslc HINTS Vulkan::glslc)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB SHADERS
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
  ${SHADER_SOURCE_DIR}/*.geom
  ${SHADER_SOURCE_DIR}/*.tesc
  ${SHADER_SOURCE_DIR}/*.tese
  ${SHADER_SOURCE_DIR}/*.mesh
  ${SHADER_SOURCE_DIR}/*.task
  ${SHADER_SOURCE_DIR}/*.rgen
  ${SHADER_SOURCE_DIR}/*.rchit
  ${SHADER_SOURCE_DIR}/*.rmiss)

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}"
)

foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME)
  add_custom_command(
    OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
    COMMAND
      ${glslc_exe}
      -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
      ${source}
    DEPENDS ${source} ${SHADER_BINARY_DIR}
    COMMENT "Compiling ${FILENAME}"
  )
  add_custom_command(
    OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.o
    COMMAND
    cd ${SHADER_BINARY_DIR} &&
    ld -r -b binary -o ${FILENAME}.o ${FILENAME}.spv
    DEPENDS ${SHADER_BINARY_DIR}/${FILENAME}.spv
    COMMENT "Converting to object ${FILENAME}"
  )
  list(APPEND SHADERS_SPV ${SHADER_BINARY_DIR}/${FILENAME}.spv)
  list(APPEND SHADERS_OBJ ${SHADER_BINARY_DIR}/${FILENAME}.o)
endforeach()

set_source_files_properties(${SHADERS_OBJ} PROPERTIES EXTERNAL_OBJECT true GENERATED true)

add_library(shaders STATIC ${SHADERS_OBJ})
set_target_properties(shaders PROPERTIES LINKER_LANGUAGE C)

# add_custom_target(spv_shaders ALL DEPENDS ${SHADERS_SPV})

