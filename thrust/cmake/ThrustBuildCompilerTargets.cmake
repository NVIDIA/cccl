# This file defines the `thrust_build_compiler_targets()` function, which
# creates the following interface targets:
#
# thrust.compiler_interface
# - Interface target linked into all targets in the thrust developer build.
#   This should not be directly used; it is only used to construct the
#   per-dialect targets below.
#
# thrust.compiler_interface_cppXX
# - Interface targets providing dialect-specific compiler flags. These should
#   be linked into the developer build targets, as they include both
#   thrust.compiler_interface and cccl.compiler_interface_cppXX.

function(thrust_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # Disabled loss-of-data conversion warnings.
    # TODO Re-enable.
    append_option_if_available("/wd4244" cxx_compile_options)

    # Disable warning about applying unary operator- to unsigned type.
    # TODO Re-enable.
    append_option_if_available("/wd4146" cxx_compile_options)
  endif()

  cccl_build_compiler_interface(
    thrust.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(thrust.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(
      thrust.compiler_interface_cpp${dialect}
      INTERFACE
        # order matters here, we need the project options to override the cccl options.
        cccl.compiler_interface_cpp${dialect}
        thrust.compiler_interface
    )
  endforeach()
endfunction()
