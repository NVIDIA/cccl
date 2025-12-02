# This file provides the following function which defines the following targets:
#
# cub.compiler_interface
# - Interface target that includes all compiler settings for cub tests, etc.

function(cub_build_compiler_targets)
  cccl_get_cub()
  cccl_get_libcudacxx()
  cccl_get_thrust()

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
      libcudacxx::libcudacxx
      CUB::CUB
      cub.thrust
  )
endfunction()
