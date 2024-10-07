# This file defines the `cudax_build_compiler_targets()` function, which
# creates the following interface targets:
#
# cudax.compiler_interface
# - Interface target linked into all targets in the CudaX developer build.
#   This should not be directly used; it is only used to construct the
#   per-dialect targets below.
#
# cudax.compiler_interface_cppXX
# - Interface targets providing dialect-specific compiler flags. These should
#   be linked into the developer build targets, as they include both
#   cudax.compiler_interface and cccl.compiler_interface_cppXX.

function(cudax_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # C4848: support for attribute 'msvc::no_unique_address' in C++17 and earlier is a vendor extension
    append_option_if_available("/wd4848" cxx_compile_options)

    # cudax requires dim3 to be usable from a constexpr context, and the CUDART headers require
    # __cplusplus to be defined for this to work:
    append_option_if_available("/Zc:__cplusplus" cxx_compile_options)
  endif()

  cccl_build_compiler_interface(cudax.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  # Clang-cuda only:
  target_compile_options(cudax.compiler_interface INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Xclang=-fcuda-allow-variadic-functions>
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno_unknown-cuda-version>
  )

  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(cudax.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(cudax.compiler_interface_cpp${dialect} INTERFACE
      # order matters here, we need the cudax options to override the cccl options.
      cccl.compiler_interface_cpp${dialect}
      cudax.compiler_interface
    )
  endforeach()

endfunction()
