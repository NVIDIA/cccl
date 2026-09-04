##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

if (libcudacxx_codegen_filecheck_simd_ptx)
  add_custom_target(libcudacxx.test.simd.ptx)
endif()
if (libcudacxx_codegen_filecheck_simd_sass)
  add_custom_target(libcudacxx.test.simd.sass)
endif()

#-----------------------------------------------------------------------------------------------------------------------
# SETUP: skip unsupported compilers, find tools, set up CUDA architectures

# GCC7 and NVCC 12.0 do not vectorize load/store.
if (
  CMAKE_CXX_COMPILER_ID STREQUAL GNU
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8
)
  message("-- Skipping simd codegen tests with GCC 7 host compiler")
  return()
endif()

if (
  CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1
)
  message("-- Skipping simd codegen tests with NVCC 12.0")
  return()
endif()

# skip nvc++
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
  message("-- Skipping simd codegen tests with NVHPC host compiler")
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  message(
    "-- clang-cuda does not support the relocatable-device-code simd codegen tests"
  )
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(simd_codegen_ptx_cuda_archs PTX)
libcudacxx_codegen_get_cuda_architectures(simd_codegen_sass_cuda_archs SASS)
if (NOT simd_codegen_ptx_cuda_archs AND NOT simd_codegen_sass_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping simd codegen tests"
  )
  return()
endif()

if (libcudacxx_codegen_filecheck_simd_ptx)
  if (simd_codegen_ptx_cuda_archs)
    add_subdirectory(load_store)
  else()
    message(
      STATUS
      "No compatible CUDA architectures requested; skipping simd PTX codegen tests"
    )
  endif()
endif()

if (libcudacxx_codegen_filecheck_simd_sass)
  if (simd_codegen_sass_cuda_archs)
    set(simd_codegen_sass_tests)
    file(
      GLOB simd_codegen_sass_tests
      CONFIGURE_DEPENDS
      "abs_diff/*.cu"
      "dot/*.cu"
      "floating_point/*.cu"
      "integer/*.cu"
      "min_max/*.cu"
    )

    # The 8-bit arithmetic and min/max tests, and the saturating-add tests, require
    # PTX ISA 9.2 paths available with CUDA 13.2 and newer.
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.2)
      list(
        REMOVE_ITEM simd_codegen_sass_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/integer/arithmetic_u8x4.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/min_max/min_max_i8x4.cu"
      )
    else()
      file(
        GLOB simd_codegen_saturating_add_tests
        CONFIGURE_DEPENDS
        "saturating_add/*.cu"
      )
      list(APPEND simd_codegen_sass_tests ${simd_codegen_saturating_add_tests})
    endif()

    libcudacxx_codegen_add_sass_tests(
      AGGREGATE_TARGET libcudacxx.test.simd.sass
      TARGET_PREFIX simd_codegen
      ARCHITECTURES ${simd_codegen_sass_cuda_archs}
      TESTS ${simd_codegen_sass_tests}
    )
  else()
    message(
      STATUS
      "No compatible CUDA architectures requested; skipping simd SASS codegen tests"
    )
  endif()
endif()
