// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This tunes the lookahead implementation of run-length encode, which is only available on SM100+. It has entirely
// different tuning parameters than the lookback implementation, so it lives in a separate file and the lookback
// implementation keeps its own tuning on older hardware architectures.

#include <cuda/__cccl_config>

#if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#  warning "This benchmark does not support being compiled for multiple architectures. Disabling it."
#else // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1

#  if __CUDA_ARCH_LIST__ < 1000
// We don't care if clang-tidy can't parse this
#    ifndef _CCCL_CLANG_TIDY_INVOKED
#      warning "Lookahead run-length encode requires at least sm_100. Disabling it."
#    endif // !defined _CCCL_CLANG_TIDY_INVOKED
#  else // __CUDA_ARCH_LIST__ < 1000

#    if __cccl_ptx_isa < 920
#      warning "Lookahead run-length encode requires at least PTX ISA 9.2. Disabling it."
#    else // if __cccl_ptx_isa < 920

#      include <nvbench_helper.cuh>

// %RANGE% TUNE_LA_IPT ipt 8:32:8
// %RANGE% TUNE_LA_CW cw 4:14:2
// %RANGE% TUNE_LA_KRS krs 3:8:1
// %RANGE% TUNE_LA_PRS prs 2:5:1
// %RANGE% TUNE_LA_PLL pll 4:8:2
// %RANGE% TUNE_LA_DPLL dpll 2:5:1
// %RANGE% TUNE_LA_DMRT dmrt 64:256:64
// %RANGE% TUNE_LA_FST fst 16:96:16

// The kernel's static_asserts reject invalid combinations at compile time (the pos ring parity bound
// 2 * prs >= krs, the thread cap, the register buffer budget). Combinations whose shared memory carve exceeds
// the device's opt-in limit compile fine but are routed to the unstaged fallback at launch, so keep krs * ipt * cw
// within the target's opt-in shared memory or the measurement reflects the fallback instead of the tuned config.

#      define USES_LOOKAHEAD() 1
// only the lookahead-viable key types: power-of-two sizes dividing 16 with alignment == size
using rle_key_types =
  nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t, int128_t>;
#      include "base.cuh"

#    endif // __cccl_ptx_isa < 920
#  endif // __CUDA_ARCH_LIST__ < 1000
#endif // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
