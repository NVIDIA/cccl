# For every public header build a TU that directly includes it
# without anything else but also pretents to be a std header
add_custom_target(libcudacxx.test.public_headers_host_only)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
else()
  find_package(CUDAToolkit)
endif()

# Grep all public headers
file(GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
)

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
  target_compile_definitions(headertest_std_${header_name} PRIVATE
    _CCCL_HEADER_TEST
    CCCL_ENABLE_ASSERTIONS
    CCCL_IGNORE_DEPRECATED_CPP_DIALECT
    CCCL_ENABLE_OPTIONAL_REF)

  # We want to ensure that we can build headers within <cuda/> with a host compiler but we need cuda_runtime_api.h
  if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    target_link_libraries(headertest_std_${header_name} NVHPC::CUDART)
  else()
    target_link_libraries(headertest_std_${header_name} CUDA::cudart)
  endif()

  add_dependencies(libcudacxx.test.public_headers_host_only headertest_std_${header_name})
endfunction()

foreach(header IN LISTS public_headers_host_only)
  libcudacxx_add_std_header_test(${header})
endforeach()
