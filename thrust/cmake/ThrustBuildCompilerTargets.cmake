# This file defines the `thrust_build_compiler_targets()` function, which
# creates the following interface targets:
#
# thrust.compiler_interface
# Provides compiler settings for all thrust tests, examples, etc. This should not be used
# directly, as it is linked to by all thrust configuration targets in THRUST_TARGETS.

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
    thrust.compiler_flags
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  add_library(thrust.compiler_interface INTERFACE)
  target_link_libraries(
    thrust.compiler_interface
    INTERFACE
      # order matters here, we need the project options to override the cccl options.
      cccl.compiler_interface
      thrust.compiler_flags
  )
endfunction()
