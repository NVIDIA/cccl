# For every public header build a TU that directly includes it
# without anything else but also pretents to be a std header
add_custom_target(libcudacxx.test.public_headers_host_only)

# Grep all public headers
file(GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

# mdspan is currently not supported on msvc outside of C++20
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "20")
  list(FILTER public_headers_host_only EXCLUDE REGEX "mdspan")
endif()

function(libcudacxx_add_std_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cpp")

  # Create the default target for that file
  set(headertest_std_${header_name} verify_${header_name})
  add_library(headertest_std_${header_name} SHARED "${headertest_src}.cpp")
  target_include_directories(headertest_std_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_options(headertest_std_${header_name} PRIVATE ${headertest_warning_levels_host})
  target_compile_definitions(headertest_std_${header_name} PRIVATE CCCL_ENABLE_ASSERTIONS)

  add_dependencies(libcudacxx.test.public_headers_host_only headertest_std_${header_name})
endfunction()

foreach(header IN LISTS public_headers_host_only)
  libcudacxx_add_std_header_test(${header})
endforeach()
