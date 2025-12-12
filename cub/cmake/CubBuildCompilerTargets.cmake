# This file provides the following function which defines the following targets:
#
# cub.compiler_interface
# - Interface target that includes all compiler settings for cub tests, etc.

function(cub_build_compiler_targets)
  cccl_get_cub()
  cccl_get_libcudacxx()
  cccl_get_thrust()

  thrust_create_target(cub.thrust HOST CPP DEVICE CUDA)

  set(ptxas_compile_options)
  if (CCCL_ENABLE_PTXAS_WARNINGS)
    list(
      APPEND ptxas_compile_options
      "--warn-on-spills"
      "--warn-on-local-memory-usage"
    )
  endif()

  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  # append ptxas compile options to cuda_compile_options with compiler specific prefix
  foreach (ptxas_compile_option ${ptxas_compile_options})
    if (
      "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA"
      OR "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVHPC"
    )
      list(APPEND cuda_compile_options "-Xptxas=${ptxas_compile_option}")
    elseif ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "CLANG")
      list(APPEND cuda_compile_options "-Xcuda-ptxas ${ptxas_compile_option}")
    endif()
  endforeach()

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
