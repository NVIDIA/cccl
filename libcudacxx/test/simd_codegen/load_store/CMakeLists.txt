##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

set(
  simd_codegen_load_store_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i8.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i32.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_bf16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f32.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i64.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f64.cu"
)

set(simd_codegen_default_prefixes SMXXX)
libcudacxx_codegen_add_ptx_tests(
  AGGREGATE_TARGET libcudacxx.test.simd.ptx
  TARGET_PREFIX simd_codegen
  ARCH 80
  CHECK_PREFIXES ${simd_codegen_default_prefixes}
  TESTS ${simd_codegen_load_store_tests}
)
libcudacxx_codegen_add_ptx_tests(
  AGGREGATE_TARGET libcudacxx.test.simd.ptx
  TARGET_PREFIX simd_codegen
  ARCH 90
  CHECK_PREFIXES ${simd_codegen_default_prefixes}
  TESTS ${simd_codegen_load_store_tests}
)

# SM100/SM120 add 32-byte vectorized checks.
if (
  CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
  AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  set(simd_codegen_sm100_prefixes SMXXX SM100-PLUS)
  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.simd.ptx
    TARGET_PREFIX simd_codegen
    ARCH 100
    CHECK_PREFIXES ${simd_codegen_sm100_prefixes}
    TESTS ${simd_codegen_load_store_tests}
  )
  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.simd.ptx
    TARGET_PREFIX simd_codegen
    ARCH 120
    CHECK_PREFIXES ${simd_codegen_sm100_prefixes}
    TESTS ${simd_codegen_load_store_tests}
  )
endif()
