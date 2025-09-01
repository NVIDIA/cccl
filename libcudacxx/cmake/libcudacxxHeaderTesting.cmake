# For every public and internal header, build a translation unit including the header
# to verify modularity and catch warnings.

add_custom_target(libcudacxx.all.headers)

if (LIBCUDACXX_ENABLE_CUDA)
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    if(MSVC)
      set(headertest_warning_levels_device -Xcompiler=/W4 -Xcompiler=/WX -Wno-deprecated-gpu-targets --use-local-env)
    else()
      set(headertest_warning_levels_device -Wall -Werror all-warnings -Wno-deprecated-gpu-targets)
    endif()
  elseif("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
    set(headertest_warning_levels_device -Wall -Werror -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions)
  else()
    set(headertest_warning_levels_device -Wall -Werror)
  endif()
endif()

if(MSVC)
  set(headertest_warning_levels_host /W4 /WX)
else()
  set(headertest_warning_levels_host -Wall -Werror)
endif()

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
  set(libcudacxx_cudart NVHPC::CUDART)
else()
  find_package(CUDAToolkit)
  set(libcudacxx_cudart CUDA::cudart)
endif()

# Internal headers
cccl_generate_header_tests(libcudacxx.headers.internal libcudacxx/include
  HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test_internal.cpp.in"
  GLOBS
    "cuda/__*/*.h"
    "cuda/std/__*/*.h"
  EXCLUDES
    "cuda/std/__cccl/prologue.h"
    "cuda/std/__cccl/epilogue.h"
    "cuda/__cuda/*"
    "cuda/std/__cuda/*"
    "cuda/std/__ptx/instructions/generated/*"
)

target_include_directories(libcudacxx.headers.internal PRIVATE "${libcudacxx_SOURCE_DIR}/include")
target_compile_options(libcudacxx.headers.internal PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${headertest_warning_levels_device}>
  $<$<COMPILE_LANGUAGE:CXX>:${headertest_warning_levels_host}>
)
target_compile_definitions(libcudacxx.headers.internal PRIVATE
  _CCCL_HEADER_TEST
  CCCL_ENABLE_ASSERTIONS
  CCCL_ENABLE_OPTIONAL_REF
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)
target_link_libraries(libcudacxx.headers.internal PRIVATE ${libcudacxx_cudart})
add_dependencies(libcudacxx.all.headers libcudacxx.headers.internal)

# Public headers (non-atomic)
set(libcudacxx_atomic_headers
  cuda/annotated_ptr
  cuda/atomic
  cuda/barrier
  cuda/latch
  cuda/pipeline
  cuda/semaphore
)

if ("${CMAKE_CUDA_COMPILER_VERSION}" MATCHES "11\\..*" OR
    "Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  list(REMOVE_ITEM libcudacxx_atomic_headers cuda/annotated_ptr)
endif()

set(libcudacxx_public_excludes ${libcudacxx_atomic_headers})

cccl_generate_header_tests(libcudacxx.headers.public libcudacxx/include
  HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test_public.cpp.in"
  GLOBS
    "cuda/*"
    "cuda/std/*"
  EXCLUDES ${libcudacxx_public_excludes}
)

target_include_directories(libcudacxx.headers.public PRIVATE "${libcudacxx_SOURCE_DIR}/include")
target_compile_options(libcudacxx.headers.public PRIVATE ${headertest_warning_levels_device})
target_compile_definitions(libcudacxx.headers.public PRIVATE
  _CCCL_HEADER_TEST
  CCCL_ENABLE_ASSERTIONS
  CCCL_IGNORE_DEPRECATED_CPP_DIALECT
  CCCL_ENABLE_OPTIONAL_REF
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)
add_dependencies(libcudacxx.all.headers libcudacxx.headers.public)

# Atomic headers require sm70+
set(architectures_at_least_sm70)
foreach(item IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(item GREATER_EQUAL 70)
    list(APPEND architectures_at_least_sm70 ${item})
  endif()
endforeach()

if (architectures_at_least_sm70 AND libcudacxx_atomic_headers)
  cccl_generate_header_tests(libcudacxx.headers.public_sm70 libcudacxx/include
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test_public.cpp.in"
    HEADERS ${libcudacxx_atomic_headers}
  )
  target_include_directories(libcudacxx.headers.public_sm70 PRIVATE "${libcudacxx_SOURCE_DIR}/include")
  target_compile_options(libcudacxx.headers.public_sm70 PRIVATE ${headertest_warning_levels_device})
  target_compile_definitions(libcudacxx.headers.public_sm70 PRIVATE
    _CCCL_HEADER_TEST
    CCCL_ENABLE_ASSERTIONS
    CCCL_IGNORE_DEPRECATED_CPP_DIALECT
    CCCL_ENABLE_OPTIONAL_REF
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
  )
  set_target_properties(libcudacxx.headers.public_sm70 PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}")
  add_dependencies(libcudacxx.all.headers libcudacxx.headers.public_sm70)
endif()

# Public headers host-only
cccl_generate_header_tests(libcudacxx.headers.host libcudacxx/include
  LANGUAGE CXX
  HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test_public.cpp.in"
  GLOBS "cuda/*"
)

target_include_directories(libcudacxx.headers.host PRIVATE "${libcudacxx_SOURCE_DIR}/include")
target_compile_options(libcudacxx.headers.host PRIVATE ${headertest_warning_levels_host})
target_compile_definitions(libcudacxx.headers.host PRIVATE
  _CCCL_HEADER_TEST
  CCCL_ENABLE_ASSERTIONS
  CCCL_IGNORE_DEPRECATED_CPP_DIALECT
  CCCL_ENABLE_OPTIONAL_REF
)
target_link_libraries(libcudacxx.headers.host PRIVATE ${libcudacxx_cudart})
add_dependencies(libcudacxx.all.headers libcudacxx.headers.host)
