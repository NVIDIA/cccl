# This file provides the following function which defines the following targets:
#
# cub.compiler_interface
# - Interface target that includes all compiler settings for cub tests, etc.

function(cub_build_compiler_targets)
  find_package(
    CUB
    REQUIRED
    CONFIG
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cub/"
  )

  find_package(
    Thrust
    ${CUB_VERSION}
    EXACT
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
  )

  thrust_set_CUB_target(CUB::CUB)
  thrust_create_target(cub.thrust HOST CPP DEVICE CUDA)

  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  cccl_build_compiler_interface(
    cub.compiler_flags
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  add_library(cub.compiler_interface INTERFACE)
  target_link_libraries(
    cub.compiler_interface
    INTERFACE
      # order matters here, we need the project options to override the cccl options.
      cccl.compiler_interface
      cub.compiler_flags
      CUB::CUB
      cub.thrust
  )
endfunction()
