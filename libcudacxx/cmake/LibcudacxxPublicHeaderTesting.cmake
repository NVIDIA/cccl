# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers)

# Grep all public headers
file(
  GLOB public_headers
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

# annotated_ptr does not work with clang cuda due to __nv_associate_access_property
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  list(FILTER public_headers EXCLUDE REGEX "annotated_ptr")
endif()

function(libcudacxx_add_public_header_test_target target_name)
  if (NOT ARGN)
    return()
  endif()

  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${ARGN}
  )

  target_compile_definitions(${target_name} PRIVATE _CCCL_HEADER_TEST)
  target_link_libraries(${target_name} PUBLIC libcudacxx.compiler_interface)
  add_dependencies(libcudacxx.test.public_headers ${target_name})
endfunction()

libcudacxx_add_public_header_test_target(
  libcudacxx.test.public_headers.base
  ${public_headers}
)
