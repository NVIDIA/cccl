# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers_host_only)

# Grep all public headers
file(GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
  # Add some files we expect to work in host only compilation
  "${libcudacxx_SOURCE_DIR}/include/cuda/bit"
  "${libcudacxx_SOURCE_DIR}/include/cuda/cmath"
  "${libcudacxx_SOURCE_DIR}/include/cuda/functional"
  "${libcudacxx_SOURCE_DIR}/include/cuda/iterator"
  "${libcudacxx_SOURCE_DIR}/include/cuda/mdspan"
  "${libcudacxx_SOURCE_DIR}/include/cuda/memory"
  "${libcudacxx_SOURCE_DIR}/include/cuda/numeric"
  "${libcudacxx_SOURCE_DIR}/include/cuda/type_traits"
  "${libcudacxx_SOURCE_DIR}/include/cuda/utility"
  "${libcudacxx_SOURCE_DIR}/include/cuda/version"
)

function(libcudacxx_create_public_header_test_host header_name, headertest_src)
  # Create the default target for that file
  set(public_headers_host_only_${header_name} verify_${header_name})
  add_library(public_headers_host_only_${header_name} SHARED "${headertest_src}.cpp")
  target_include_directories(public_headers_host_only_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_definitions(public_headers_host_only_${header_name} PRIVATE _CCCL_HEADER_TEST)

  # Bring in the global CCCL compile definitions
  target_link_libraries(public_headertest_${header_name} PUBLIC libcudacxx.compiler_interface)

  add_dependencies(libcudacxx.test.public_headers_host_only public_headers_host_only_${header_name})
endfunction()

function(libcudacxx_add_public_headers_host_only header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in" "${headertest_src}.cpp")

  # Create the default target for that file
  libcudacxx_create_public_header_test_host(${header_name}, ${headertest_src})
endfunction()

foreach(header IN LISTS public_headers_host_only)
  libcudacxx_add_public_headers_host_only(${header})
endforeach()
