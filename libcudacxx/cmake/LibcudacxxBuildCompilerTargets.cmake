# This file defines the `libcudacxx_build_compiler_targets()` function, which
# creates the following interface targets:
#
# libcudacxx.compiler_interface
# - Interface target linked into all targets in the libcudacxx developer build.
#   Defines common warning flags, definitions, etc, including those defined in
#   the global CCCL targets.

function(libcudacxx_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  #  if (CCCL_USE_LIBCXX)
  #    list(APPEND cxx_compile_options "-stdlib=libc++")
  #    list(APPEND cxx_compile_definitions "_ALLOW_UNSUPPORTED_LIBCPP=1")
  #  endif()

  # Set test specific flags
  list(APPEND cxx_compile_definitions "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
  list(APPEND cxx_compile_definitions "CCCL_ENABLE_ASSERTIONS")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_CPP_DIALECT")
  list(APPEND cxx_compile_definitions "CCCL_ENABLE_OPTIONAL_REF")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_DISCARD_MEMORY_HEADER")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER")

  cccl_build_compiler_interface(libcudacxx.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  # libcudacxx only builds a single dialect at a time, so link to the currently
  # selected dialect target from cccl:
  target_link_libraries(libcudacxx.compiler_interface INTERFACE
    cccl.compiler_interface_cpp${CMAKE_CUDA_STANDARD}
  )
 endfunction()
