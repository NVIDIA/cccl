// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/__functional/operations.h>

// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1
// %RANGE% TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK ltpb 128:1024:32
// %RANGE% TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD lipt 7:24:1

// Without a max segment size guarantee only the large policy is used; pin the small and medium ones to the defaults.
#define TUNE_S_THREADS_PER_WARP            1
#define TUNE_M_THREADS_PER_WARP            32
#define TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD 16
#define TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD 16

using op_t = ::cuda::std::plus<>;

#include "variable_base.cuh"
