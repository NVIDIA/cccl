// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1
// %RANGE% TUNE_S_THREADS_PER_WARP stpw 1:32:1
// %RANGE% TUNE_M_THREADS_PER_WARP mtpw 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK ltpb 128:1024:32
// %RANGE% TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD sipt 1:32:1
// %RANGE% TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD mipt 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD lipt 7:24:1

using value_types = all_types;
using op_t        = ::cuda::std::plus<>;
#include "base.cuh"
