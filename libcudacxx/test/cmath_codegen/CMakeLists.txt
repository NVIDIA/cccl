##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.cmath.ptx)

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
  message("-- Skipping cmath PTX codegen tests without NVCC")
  return()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
  message("-- Skipping cmath PTX codegen tests with NVHPC host compiler")
  return()
endif()

file(GLOB libcudacxx_cmath_codegen_tests CONFIGURE_DEPENDS "*.cu")
if (NOT libcudacxx_cmath_codegen_tests)
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(cmath_codegen_cuda_archs PTX)
if (NOT cmath_codegen_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping cmath PTX codegen tests"
  )
  return()
endif()

foreach (arch IN LISTS cmath_codegen_cuda_archs)
  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.cmath.ptx
    TARGET_PREFIX cmath_codegen
    ARCH "${arch}"
    CHECK_PREFIXES SMXX
    TESTS ${libcudacxx_cmath_codegen_tests}
  )
endforeach()
