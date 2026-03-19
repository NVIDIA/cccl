# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

cccl_get_cudatoolkit()

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

function(
  libcudacxx_add_public_header_test_host_target
  target_name
  parent_target
  with_ctk
)
  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CXX
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${public_headers_host_only}
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE #
      ${public_host_header_cxx_compile_definitions}
      _CCCL_HEADER_TEST
  )
  target_compile_options(
    ${target_name}
    PRIVATE ${public_host_header_cxx_compile_options}
  )
  target_link_libraries(${target_name} PUBLIC libcudacxx.compiler_interface)
  if (with_ctk)
    target_link_libraries(${target_name} PUBLIC CUDA::cudart)
  endif()
  add_dependencies(${parent_target} ${target_name})
endfunction()

libcudacxx_add_public_header_test_host_target(
  libcudacxx.test.public_headers_host_only.base
  libcudacxx.test.public_headers_host_only
  OFF
)
libcudacxx_add_public_header_test_host_target(
  libcudacxx.test.public_headers_host_only_with_ctk.base
  libcudacxx.test.public_headers_host_only_with_ctk
  ON
)
