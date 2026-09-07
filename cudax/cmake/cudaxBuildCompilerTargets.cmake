# Including this file defines the following targets:
#
# cudax.compiler_interface
# - Interface target that includes all compiler settings for cudax tests, etc.

cccl_get_cub()
cccl_get_cudax()
cccl_get_libcudacxx()
cccl_get_thrust()

set(cuda_compile_options)
set(cxx_compile_options)
set(cxx_compile_definitions)

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # C4848: support for attribute 'msvc::no_unique_address' in C++17 and earlier is a vendor extension
  append_option_if_available("/wd4848" cxx_compile_options)

  # XXX Temporary hack for STF !
  # C4267: conversion from 'meow' to 'purr', possible loss of data
  append_option_if_available("/wd4267" cxx_compile_options)

  # C4459 : declaration of 'identifier' hides global declaration
  # We work around std::chrono::last which hides some internal "last" variable
  append_option_if_available("/wd4459" cxx_compile_options)

  # stf used getenv which is potentially unsafe but not in our context
  list(APPEND cxx_compile_definitions "_CRT_SECURE_NO_WARNINGS")
endif()

if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # stf heavily uses host device lambdas which break on clang due to a warning about the implicitly
  # deleted copy constructor
  # TODO(bgruber): remove this when NVBug 4980157 is resolved
  append_option_if_available("-Wno-deprecated-copy" cxx_compile_options)
endif()

list(APPEND cxx_compile_definitions CCCL_ENABLE_ASSERTIONS)

# Some groups related experimental code is located directly in libcu++ and is guarded by
# _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX macro, otherwise it would lead to a lot of code duplication. We define this
# macro for cudax code globally, to get access to get access to the code.
#
# Can be removed once groups are no longer experimental.
list(APPEND cxx_compile_definitions _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX)

cccl_build_compiler_interface(
  cudax.compiler_flags
  "${cuda_compile_options}"
  "${cxx_compile_options}"
  "${cxx_compile_definitions}"
)

add_library(cudax.compiler_interface INTERFACE)
target_link_libraries(
  cudax.compiler_interface
  INTERFACE
    # order matters here, we need the cudax options to override the cccl options.
    cccl.compiler_interface
    cudax.compiler_flags
    libcudacxx::libcudacxx
    CUB::CUB
    Thrust::Thrust
    cudax::cudax
)
