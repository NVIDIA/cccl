# This file defines the `libcudacxx_build_compiler_targets()` function, which
# creates the following interface targets:
#
# libcudacxx.compiler_interface
# - Interface target linked into all targets in the CUB developer build.
#   This should not be directly used; it is only used to construct the
#   per-dialect targets below.
#
# libcudacxx.compiler_interface_cppXX
# - Interface targets providing dialect-specific compiler flags. These should
#   be linked into the developer build targets, as they include both
#   libcudacxx.compiler_interface and cccl.compiler_interface_cppXX.

function(libcudacxx_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  # Set test specific flags
  list(APPEND cxx_compile_definitions "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
  list(APPEND cxx_compile_definitions "CCCL_ENABLE_ASSERTIONS")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_CPP_DIALECT")
  list(APPEND cxx_compile_definitions "CCCL_ENABLE_OPTIONAL_REF")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_DISCARD_MEMORY_HEADER")

  cccl_build_compiler_interface(libcudacxx.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(libcudacxx.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(libcudacxx.compiler_interface_cpp${dialect} INTERFACE
      # order matters here, we need the project options to override the cccl options.
      cccl.compiler_interface_cpp${dialect}
      libcudacxx.compiler_interface
    )
  endforeach()
endfunction()
