##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.atomics.sass)

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
  message("-- Skipping atomic SASS codegen tests without NVCC")
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(atomic_codegen_sass_cuda_archs SASS)
if (NOT atomic_codegen_sass_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping atomic SASS codegen tests"
  )
  return()
endif()

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1)
  set(atomic_codegen_sass_cuda_version_prefix CUDA12-0)
else()
  set(atomic_codegen_sass_cuda_version_prefix CUDA12-1-PLUS)
endif()

set(libcudacxx_atomic_codegen_tests)
if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  file(GLOB libcudacxx_atomic_codegen_tests CONFIGURE_DEPENDS "*.cu")
endif()

if (NOT libcudacxx_atomic_codegen_tests)
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

include(CheckSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
list(APPEND CMAKE_REQUIRED_INCLUDES "${libcudacxx_SOURCE_DIR}/include")
check_source_compiles(
  CUDA
  [[
    #include <cuda/std/__cccl/extended_data_types.h>

    #if !_CCCL_HAS_INT128()
    #  error "128-bit integers are not supported"
    #endif

    int main() { return 0; }
  ]]
  libcudacxx_atomic_codegen_has_int128
)
check_source_compiles(
  CUDA
  [[
    #include <cuda/std/detail/__config>

    #if defined(__CUDA_ARCH__) \
      && (!_CCCL_CTK_AT_LEAST(13, 5) || !_CCCL_HAS_NV_ATOMIC_BUILTINS())
    #  error "the NVVM atomic backend is unavailable"
    #endif

    __global__ void test() {}
    int main() { test<<<1, 1>>>(); }
  ]]
  libcudacxx_atomic_codegen_uses_nvvm_backend
)
cmake_pop_check_state()

if (libcudacxx_atomic_codegen_uses_nvvm_backend)
  set(atomic_codegen_sass_backend_prefix NVVM)
else()
  set(atomic_codegen_sass_backend_prefix PTX)
endif()

if (NOT libcudacxx_atomic_codegen_has_int128)
  message(
    STATUS
    "Skipping 128-bit atomic SASS codegen tests: 128-bit integers are not supported"
  )
  list(FILTER libcudacxx_atomic_codegen_tests EXCLUDE REGEX "_types_128_")
else()
  # Native 128-bit atomic_ref operations require the PTX ISA 8.4 path available
  # with CUDA 12.4 and newer.
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    list(
      FILTER libcudacxx_atomic_codegen_tests
      EXCLUDE
      REGEX "_types_128_atomic_ref[.]cu$"
    )
  endif()
endif()

# Restrict FileCheck to the wrapper so instructions in outlined helpers cannot
# mask an inlining regression.
libcudacxx_codegen_add_sass_tests(
  AGGREGATE_TARGET libcudacxx.test.atomics.sass
  TARGET_PREFIX atomic_codegen
  ARCHITECTURES ${atomic_codegen_sass_cuda_archs}
  DUMP_FUNCTIONS atomic_codegen_test
  CHECK_PREFIXES
    ${atomic_codegen_sass_cuda_version_prefix}
    ${atomic_codegen_sass_backend_prefix}
  TESTS ${libcudacxx_atomic_codegen_tests}
  COMPILE_DEFINITIONS _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1
)
