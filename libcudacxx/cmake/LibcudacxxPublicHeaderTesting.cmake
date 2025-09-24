# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers)

# Grep all public headers
file(GLOB public_headers
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

# We need to handle atomic headers differently as they do not compile on architectures below sm70
set(architectures_at_least_sm70)
foreach(item IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(item GREATER_EQUAL 70)
    list(APPEND architectures_at_least_sm70 ${item})
  endif()
endforeach()

function(libcudacxx_create_public_header_test header_name, headertest_src)
  # Create the default target for that file
  set(public_headertest_${header_name} verify_${header_name})
  add_library(public_headertest_${header_name} SHARED "${headertest_src}.cu")
  target_include_directories(public_headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_definitions(public_headertest_${header_name} PRIVATE _CCCL_HEADER_TEST)

  # Bring in the global CCCL compile definitions
  target_link_libraries(public_headertest_${header_name} PUBLIC libcudacxx.compiler_interface)

  # Ensure that if this is an atomic header, we only include the right architectures
  string(REGEX MATCH "atomic|barrier|latch|semaphore|annotated_ptr|pipeline" match "${header}")
  if(match)
    # Ensure that we only compile the header when we have some architectures enabled
    if (NOT architectures_at_least_sm70)
      return()
    endif()
    set_target_properties(public_headertest_${header_name} PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}")
  endif()

  add_dependencies(libcudacxx.test.public_headers public_headertest_${header_name})
endfunction()

function(libcudacxx_add_public_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in" "${headertest_src}.cu")

  # Create the default target for that file
  libcudacxx_create_public_header_test(${header_name}, ${headertest_src})
endfunction()

foreach(header IN LISTS public_headers)
  libcudacxx_add_public_header_test(${header})
endforeach()
