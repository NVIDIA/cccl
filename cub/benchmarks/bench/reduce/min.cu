// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// This benchmark is intended to cover DPX instructions on Hopper+ architectures. It specifically uses cuda::minimum<>
// instead of a user-defined operator, which CUB recognizes to select an optimized code path.

// Tuning parameters found for ::cuda::minimum<> apply equally for ::cuda::maximum<>
// Tuning parameters found for signed integer types apply equally for unsigned integer types
// TODO(bgruber): do tuning parameters found for int16_t apply equally for __half or __nv_bfloat16 on SM90+?

#include <cuda/functional>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

// __half and __nv_bfloat16 are appended for full (non-tuning) runs; CUB has fast paths for them (see #9587).
using value_types =
  push_back_t<fundamental_types
#if !defined(TUNE_T) && _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
              ,
              __half
#endif
#if !defined(TUNE_T) && _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
              ,
              __nv_bfloat16
#endif
              >;
using op_t = ::cuda::minimum<>;
#include "base.cuh"
