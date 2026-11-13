##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.random.sass)

if (
  NOT CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA
  OR CMAKE_CXX_COMPILER_ID STREQUAL NVHPC
)
  message(
    "-- Skipping random SASS codegen tests without NVCC and a supported host compiler"
  )
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(random_codegen_cuda_archs SASS)
if (NOT random_codegen_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping random SASS codegen tests"
  )
  return()
endif()

file(GLOB random_codegen_tests CONFIGURE_DEPENDS "*.cu")

libcudacxx_codegen_add_sass_tests(
  AGGREGATE_TARGET libcudacxx.test.random.sass
  TARGET_PREFIX random_codegen
  ARCHITECTURES ${random_codegen_cuda_archs}
  TESTS ${random_codegen_tests}
)
