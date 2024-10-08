# For every private internal header build a TU that directly includes it
# without anything else to verify whether its properly modular.
# We also want to check whether all potential fallbacks are actually valid,
# even if we never exercise them in the current support matrix
add_custom_target(libcudacxx.test.internal_headers)

find_package(CUDAToolkit)

# Grep all internal headers
file(GLOB_RECURSE internal_headers
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/cuda/"
  CONFIGURE_DEPENDS
  ${libcudacxx_SOURCE_DIR}/include/cuda/__*/*.h
  ${libcudacxx_SOURCE_DIR}/include/cuda/std/__*/*.h
)

# headers in `__cuda` are meant to come after the related "cuda" headers so they do not compile on their own
list(FILTER internal_headers EXCLUDE REGEX "__cuda/*")

# mdspan is currently not supported on msvc outside of C++20
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "20")
  list(FILTER internal_headers EXCLUDE REGEX "mdspan")
endif()

function(libcudacxx_create_internal_header_test header_name, headertest_src, fallback)
  if(fallback)
    set(header_name "${header_name}_fallback")
  endif()

  set(headertest_${header_name} verify_${header_name})
  # cuda internal headers should not be tested against a host compiler
  if(header_name MATCHES "__ptx.*")
    add_library(headertest_${header_name} SHARED "${headertest_src}.cu")
  else()
    add_library(headertest_${header_name} SHARED "${headertest_src}.cu" "${headertest_src}.cpp")
  endif()

  target_include_directories(headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_options(headertest_${header_name}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${headertest_warning_levels_device}>
    $<$<COMPILE_LANGUAGE:CXX>:${headertest_warning_levels_host}>
    -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
  target_compile_definitions(headertest_${header_name} PRIVATE CCCL_ENABLE_ASSERTIONS)
  target_link_libraries(headertest_${header_name} CUDA::cudart)

  if(fallback)
    target_compile_definitions(headertest_${header_name} PRIVATE "-D${fallback}")
  endif()
  add_dependencies(libcudacxx.test.internal_headers headertest_${header_name})
endfunction()

function(libcudacxx_add_internal_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cpp")

  # Create the default target for that file
  libcudacxx_create_internal_header_test(${header_name}, ${headertest_src}, "")

  # MSVC cannot handle some of the fallbacks
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if("${header}" MATCHES "is_base_of" OR
       "${header}" MATCHES "is_nothrow_destructible" OR
       "${header}" MATCHES "is_polymorphic")
      return()
    endif()
  endif()

  # Seach the file for a fallback definition
  file(READ ${libcudacxx_SOURCE_DIR}/include/cuda/${header} header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if(fallback)
    libcudacxx_create_internal_header_test(${header_name}, ${headertest_src}, ${fallback})
  endif()
endfunction()

foreach(header IN LISTS internal_headers)
  libcudacxx_add_internal_header_test(${header})
endforeach()
