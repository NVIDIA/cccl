# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.internal_headers)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
else()
  find_package(CUDAToolkit)
endif()

# Grep all internal headers
file(GLOB_RECURSE internal_headers
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  ${libcudacxx_SOURCE_DIR}/include/cuda/__*/*.h
  ${libcudacxx_SOURCE_DIR}/include/cuda/std/__*/*.h
)

# Exclude <cuda/std/__cccl/(prologue|epilogue|visibility).h> from the test
list(FILTER internal_headers EXCLUDE REGEX "__cccl/(prologue|epilogue|visibility)\.h")

# headers in `__cuda` are meant to come after the related "cuda" headers so they do not compile on their own
list(FILTER internal_headers EXCLUDE REGEX "__cuda/*")

# generated cuda::ptx headers are not standalone
list(FILTER internal_headers EXCLUDE REGEX "__ptx/instructions/generated")

function(libcudacxx_create_internal_header_test header_name, headertest_src)
  # Create the default target for that file
  set(internal_headertest_${header_name} verify_${header_name})
  add_library(internal_headertest_${header_name} SHARED "${headertest_src}.cu")
  target_include_directories(internal_headertest_${header_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_definitions(internal_headertest_${header_name} PRIVATE _CCCL_HEADER_TEST)

  # Bring in the global CCCL compile definitions
  # Link against the right cuda runtime
  if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    target_link_libraries(internal_headertest_${header_name} PUBLIC
      libcudacxx.compiler_interface
      NVHPC::CUDART
    )
  else()
    target_link_libraries(internal_headertest_${header_name} PUBLIC
      libcudacxx.compiler_interface
      CUDA::cudart
    )
  endif()

  # Ensure that if this is an atomic header, we only include the right architectures
  string(REGEX MATCH "atomic|barrier|latch|semaphore|annotated_ptr|pipeline" match "${header}")
  if(match)
    # Ensure that we only compile the header when we have some architectures enabled
    if (NOT architectures_at_least_sm70)
      return()
    endif()
    set_target_properties(internal_headertest_${header_name} PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}")
  endif()

  add_dependencies(libcudacxx.test.internal_headers internal_headertest_${header_name})
endfunction()

# We have fallbacks for some type traits that we want to explicitly test so that they do not bitrot
function(libcudacxx_create_internal_header_fallback_test header_name, headertest_src)
  # MSVC cannot handle some of the fallbacks
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if("${header}" MATCHES "is_base_of" OR
       "${header}" MATCHES "is_nothrow_destructible" OR
       "${header}" MATCHES "is_polymorphic")
      return()
    endif()
  endif()

  # Search the file for a fallback definition
  file(READ ${libcudacxx_SOURCE_DIR}/include/${header} header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if(fallback)
    # Adopt the filename for the fallback tests
    set(header_name "${header_name}_fallback")
    libcudacxx_create_internal_header_test(${header_name}, ${headertest_src})
    target_compile_definitions(internal_headertest_${header_name} PRIVATE "-D${fallback}")
  endif()
endfunction()

function(libcudacxx_add_internal_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in" "${headertest_src}.cu")

  # Create the default target for that file
  libcudacxx_create_internal_header_test(${header_name}, ${headertest_src})

  # Optionally create a fallback target for that file
  libcudacxx_create_internal_header_fallback_test(${header_name}, ${headertest_src})
endfunction()

foreach(header IN LISTS internal_headers)
  libcudacxx_add_internal_header_test(${header})
endforeach()
