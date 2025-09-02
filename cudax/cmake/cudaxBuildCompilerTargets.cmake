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

find_package(Thrust CONFIG REQUIRED
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
)

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

    # XXX Temporary hack for STF !
    # C4267: conversion from 'meow' to 'purr', possible loss of data
    append_option_if_available("/wd4267" cxx_compile_options)

    # C4459 : declaration of 'identifier' hides global declaration
    # We work around std::chrono::last which hides some internal "last" variable
    append_option_if_available("/wd4459" cxx_compile_options)

    # stf used getenv which is potentially unsafe but not in our context
    list(APPEND cxx_compile_definitions "_CRT_SECURE_NO_WARNINGS")
  endif()

  if("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # stf heavily uses host device lambdas which break on clang due to a warning about the implicitly
    # deleted copy constructor
    # TODO(bgruber): remove this when NVBug 4980157 is resolved
    append_option_if_available("-Wno-deprecated-copy" cxx_compile_options)
  endif()

  cccl_build_compiler_interface(cudax.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  # Ensure that we test with assertions enabled
  target_compile_definitions(cudax.compiler_interface INTERFACE CCCL_ENABLE_ASSERTIONS)

  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(cudax.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(cudax.compiler_interface_cpp${dialect} INTERFACE
      # order matters here, we need the cudax options to override the cccl options.
      cccl.compiler_interface_cpp${dialect}
      cudax.compiler_interface
      libcudacxx::libcudacxx
      CUB::CUB
      Thrust::Thrust
    )
  endforeach()

endfunction()
