// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This tunes the warpspeed implementation of scan, which is only available on SM100+. It has entirely different tuning
// parameters and is agnostic of the offset type. It is thus in a separate file, so we can continue to tune the old scan
// implementation on older hardware architectures.

#include <cuda/__cccl_config>

#if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#  warning "This benchmark does not support being compiled for multiple architectures. Disabling it."
#else // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1

#  if __CUDA_ARCH_LIST__ < 1000
#    warning "Warpspeed scan requires at least sm_100. Disabling it."
#  else // __CUDA_ARCH_LIST__ < 1000

#    if __cccl_ptx_isa < 860
#      warning "Warpspeed scan requires at least PTX ISA 8.6. Disabling it."
#    else // if __cccl_ptx_isa < 860

#      include <nvbench_helper.cuh>

// %RANGE% TUNE_NUM_REDUCE_SCAN_WARPS wrps 1:8:1
// %RANGE% TUNE_NUM_LOOKBACK_ITEMS lbi 1:8:1

// TODO(bgruber): find a good range and step width, items per thread should be coprime with 32 to avoid SMEM conflicts.
// Should we specify nominal items per thread instead?
// %RANGE% TUNE_ITEMS_PLUS_ONE ipt 8:256:8

#      define USES_WARPSPEED() 1
using op_t              = ::cuda::std::plus<>;
using scan_offset_types = nvbench::type_list<int64_t>;
#      include "base.cuh"

#    endif // __cccl_ptx_isa < 860
#  endif // __CUDA_ARCH_LIST__ < 1000
#endif // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
