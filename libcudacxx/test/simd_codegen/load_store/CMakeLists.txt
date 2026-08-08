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

foreach (arch IN LISTS simd_codegen_cuda_archs)
  set(simd_codegen_check_prefixes SMXXX)
  if (
    arch MATCHES "^([0-9]+)"
    AND CMAKE_MATCH_1 GREATER_EQUAL 100
    AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
    AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
  )
    list(APPEND simd_codegen_check_prefixes SM100-PLUS)
  endif()

  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.simd.ptx
    TARGET_PREFIX simd_codegen
    ARCH "${arch}"
    CHECK_PREFIXES ${simd_codegen_check_prefixes}
    TESTS ${simd_codegen_load_store_tests}
  )
endforeach()
