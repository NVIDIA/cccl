# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

cccl_get_cudatoolkit()

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.internal_headers)

# Grep all internal headers
file(
  GLOB_RECURSE internal_headers
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  ${libcudacxx_SOURCE_DIR}/include/cuda/__*/*.h
  ${libcudacxx_SOURCE_DIR}/include/cuda/std/__*/*.h
)

# Exclude <cuda/std/__cccl/(prologue|epilogue|visibility).h> from the test
list(
  FILTER internal_headers
  EXCLUDE
  REGEX "__cccl/(prologue|epilogue|visibility)\.h"
)

# headers in `__cuda` are meant to come after the related "cuda" headers so they do not compile on their own
list(FILTER internal_headers EXCLUDE REGEX "__cuda/*")

# generated cuda::ptx headers are not standalone
list(FILTER internal_headers EXCLUDE REGEX "__ptx/instructions/generated")

function(libcudacxx_add_internal_header_test_target target_name)
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
  target_link_libraries(
    ${target_name}
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
  add_dependencies(libcudacxx.test.internal_headers ${target_name})
endfunction()

libcudacxx_add_internal_header_test_target(
  libcudacxx.test.internal_headers.base
  ${internal_headers}
)

# We have fallbacks for some type traits that we want to explicitly test so that they do not bitrot.
set(internal_headers_fallback)
set(internal_headers_fallback_per_header_defines)
foreach (header IN LISTS internal_headers)
  # MSVC cannot handle some of the fallbacks.
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (
      "${header}" MATCHES "is_base_of"
      OR "${header}" MATCHES "is_nothrow_destructible"
      OR "${header}" MATCHES "is_polymorphic"
    )
      continue()
    endif()
  endif()

  file(READ "${libcudacxx_SOURCE_DIR}/include/${header}" header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if (fallback)
    list(APPEND internal_headers_fallback "${header}")
    string(
      REGEX REPLACE
      "([][+.*^$()|?\\\\])"
      "\\\\\\1"
      header_regex
      "${header}"
    )
    list(
      APPEND internal_headers_fallback_per_header_defines
      DEFINE
      "${fallback}"
      "^${header_regex}$"
    )
  endif()
endforeach()

if (internal_headers_fallback)
  cccl_generate_header_tests(
    libcudacxx.test.internal_headers.fallback
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${internal_headers_fallback}
    PER_HEADER_DEFINES ${internal_headers_fallback_per_header_defines}
  )
  target_compile_definitions(
    libcudacxx.test.internal_headers.fallback
    PRIVATE _CCCL_HEADER_TEST
  )
  target_link_libraries(
    libcudacxx.test.internal_headers.fallback
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
  add_dependencies(
    libcudacxx.test.internal_headers
    libcudacxx.test.internal_headers.fallback
  )
endif()
