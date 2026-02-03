# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers_host_only)
add_custom_target(libcudacxx.test.public_headers_host_only_with_ctk)

# Grep all public headers
file(
  GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

set(public_host_header_cxx_compile_options)
set(public_host_header_cxx_compile_definitions)

# Specifically add libc++ testing if requested to the libcudacxx host suite
if (CCCL_USE_LIBCXX)
  list(APPEND public_host_header_cxx_compile_options "-stdlib=libc++")
endif()

function(libcudacxx_create_public_header_test_host header_name headertest_src)
  # Create the default target for that file
  add_library(
    public_headers_host_only_${header_name}
    SHARED
    "${headertest_src}.cpp"
  )
  cccl_configure_target(public_headers_host_only_${header_name})
  target_compile_definitions(
    public_headers_host_only_${header_name}
    PRIVATE #
      ${public_host_header_cxx_compile_definitions}
      _CCCL_HEADER_TEST
  )
  target_compile_options(
    public_headers_host_only_${header_name}
    PRIVATE ${public_host_header_cxx_compile_options}
  )
  target_link_libraries(
    public_headers_host_only_${header_name}
    PUBLIC libcudacxx.compiler_interface
  )
  add_dependencies(
    libcudacxx.test.public_headers_host_only
    public_headers_host_only_${header_name}
  )
endfunction()

function(
  libcudacxx_create_public_header_test_host_with_ctk
  header_name
  headertest_src
)
  # Create the default target for that file
  add_library(
    public_headers_host_only_with_ctk_${header_name}
    SHARED
    "${headertest_src}.cpp"
  )
  cccl_configure_target(public_headers_host_only_with_ctk_${header_name})
  target_compile_definitions(
    public_headers_host_only_with_ctk_${header_name}
    PRIVATE #
      ${public_host_header_cxx_compile_definitions}
      _CCCL_HEADER_TEST
  )
  target_compile_options(
    public_headers_host_only_with_ctk_${header_name}
    PRIVATE ${public_host_header_cxx_compile_options}
  )
  target_link_libraries(
    public_headers_host_only_with_ctk_${header_name}
    PUBLIC libcudacxx.compiler_interface CUDA::cudart
  )
  add_dependencies(
    libcudacxx.test.public_headers_host_only_with_ctk
    public_headers_host_only_with_ctk_${header_name}
  )
endfunction()

function(libcudacxx_add_public_headers_host_only header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in"
    "${headertest_src}.cpp"
  )

  # Create the default target for that file
  libcudacxx_create_public_header_test_host(${header_name} ${headertest_src})
  libcudacxx_create_public_header_test_host_with_ctk(${header_name} ${headertest_src})
endfunction()

foreach (header IN LISTS public_headers_host_only)
  libcudacxx_add_public_headers_host_only(${header})
endforeach()
