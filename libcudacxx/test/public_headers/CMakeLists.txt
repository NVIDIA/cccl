# For every public header build a TU that directly includes it
# without anything else to verify whether its properly modular
add_custom_target(libcudacxx.test.public_headers)

# Grep all public headers
file(GLOB public_headers
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

# annotated_ptr does not work with nvcc 11.1 or clang cuda due to __nv_associate_access_property
if ("${CMAKE_CUDA_COMPILER_VERSION}" MATCHES "11\.*\.*" OR
    "${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
  list(FILTER public_headers EXCLUDE REGEX "annotated_ptr")
endif()

# mdspan is currently not supported on msvc outside of C++20
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "20")
  list(FILTER public_headers EXCLUDE REGEX "mdspan")
endif()

# We need to handle atomic headers differently as they do not compile on architectures below sm70
set(architectures_at_least_sm70)
foreach(item IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(item GREATER_EQUAL 70)
    list(APPEND architectures_at_least_sm70 ${item})
  endif()
endforeach()

function(libcudacxx_add_public_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cu")

  # Create the default target for that file
  set(headertest_${header_name} verify_${header_name})
  add_library(headertest_${header_name} SHARED "${headertest_src}.cu")
  target_include_directories(headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_options(headertest_${header_name}
                         PRIVATE
                         ${headertest_warning_levels_device}
                         -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
  target_compile_definitions(headertest_${header_name} PRIVATE CCCL_ENABLE_ASSERTIONS)

  # Ensure that if this is an atomic header, we only include the right architectures
  string(REGEX MATCH "atomic|barrier|latch|semaphore|annotated_ptr|pipeline" match "${header}")
  if(match)
    # Ensure that we only compile the header when we have some architectures enabled
    if (NOT architectures_at_least_sm70)
      return()
    endif()
    set_target_properties(headertest_${header_name} PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}")
  endif()

  add_dependencies(libcudacxx.test.public_headers headertest_${header_name})
endfunction()

foreach(header IN LISTS public_headers)
  libcudacxx_add_public_header_test(${header})
endforeach()
