if (NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  return()
endif()

# sm_107 family-specific features are supported since 13.4
if ("${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "13.4")
  return()
endif()

set_directory_properties(
  PROPERTIES INCLUDE_DIRECTORIES "${libcudacxx_SOURCE_DIR}/include"
)

set(target_name "libcudacxx.test.nvtarget.family_specific")
add_library(${target_name} OBJECT family_specific.cu)
set_target_properties(
  ${target_name}
  PROPERTIES CUDA_ARCHITECTURES "107f-virtual"
)
add_dependencies(libcudacxx.test.nvtarget ${target_name})
