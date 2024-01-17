# For every public header build a TU that directly includes it
# without anything else but also pretents to be a std header
add_custom_target(libcudacxx.test.std_headers)

# Grep all public headers
file(GLOB std_headers
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/cuda/std"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

function(libcudacxx_add_std_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cpp")

  # Create the default target for that file
  set(headertest_${header_name} verify_${header_name})
  add_library(headertest_${header_name} SHARED "${headertest_src}.cpp")
  target_include_directories(headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_include_directories(headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include/cuda/std/detail/libcxx/include")
  target_compile_options(headertest_${header_name} PRIVATE ${headertest_warning_levels_host})

  add_dependencies(libcudacxx.test.std_headers headertest_${header_name})
endfunction()

foreach(header IN LISTS std_headers)
  libcudacxx_add_std_header_test(${header})
endforeach()
