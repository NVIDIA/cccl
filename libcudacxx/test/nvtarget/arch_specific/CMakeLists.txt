if (NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  return()
endif()

# arch-specific features are supported since 12.9
if ("${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "12.9")
  return()
endif()

set_directory_properties(
  PROPERTIES INCLUDE_DIRECTORIES "${libcudacxx_SOURCE_DIR}/include"
)

set(target_name "libcudacxx.test.nvtarget.arch_specific")
add_library(${target_name} OBJECT arch_specific.cu)
set_target_properties(
  ${target_name}
  PROPERTIES CUDA_ARCHITECTURES "103a-virtual"
)
add_dependencies(libcudacxx.test.nvtarget ${target_name})
