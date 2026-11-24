##===----------------------------------------------------------------------===##
##
## Part of CUDA Experimental in CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(cudax.test.random_codegen.sass ALL)

if (
  NOT CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA
  OR CMAKE_CXX_COMPILER_ID STREQUAL NVHPC
)
  message(
    FATAL_ERROR
    "The random SASS tests require NVCC and a supported host compiler."
  )
endif()

include("${CCCL_SOURCE_DIR}/libcudacxx/test/cmake/CodegenTest.cmake")
libcudacxx_codegen_check_tools(codegen_tools_available)
if (NOT codegen_tools_available)
  message(
    FATAL_ERROR
    "The random SASS tests require FileCheck, cuobjdump, and bash."
  )
endif()

libcudacxx_codegen_get_cuda_architectures(random_codegen_cuda_archs SASS)
if (NOT random_codegen_cuda_archs)
  message(
    FATAL_ERROR
    "The random SASS tests require at least one real CUDA architecture."
  )
endif()

link_libraries(cudax::cudax)

libcudacxx_codegen_add_sass_tests(
  AGGREGATE_TARGET cudax.test.random_codegen.sass
  TARGET_PREFIX cudax_random_codegen
  ARCHITECTURES ${random_codegen_cuda_archs}
  TESTS "${CMAKE_CURRENT_SOURCE_DIR}/mwc_single_step.cu"
)
