// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda/__cccl_config>

#ifndef TEST_HALF_T
#  if _CCCL_HAS_NVFP16()
#    define TEST_HALF_T() 1
#  else
#    define TEST_HALF_T() 0
#  endif
#endif // TEST_HALF_T

#ifndef TEST_BF_T
#  if _CCCL_HAS_NVBF16()
#    define TEST_BF_T() 1
#  else
#    define TEST_BF_T() 0
#  endif
#endif // TEST_BF_T

#ifndef TEST_INT128
#  if _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG) // clang-cuda crashes with int128 in generator.cu
#    define TEST_INT128() 1
#  else
#    define TEST_INT128() 0
#  endif
#endif // TEST_INT128

#if TEST_HALF_T()
#  include <cuda_fp16.h>

#  include <c2h/half.cuh>
#endif // TEST_HALF_T()

#if TEST_BF_T()
#  include <cuda_bf16.h>

#  include <c2h/bfloat16.cuh>
#endif // TEST_BF_T()
