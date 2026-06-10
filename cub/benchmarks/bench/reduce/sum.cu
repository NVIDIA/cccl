// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// This benchmark is intended to cover redux instructions on Ampere+ architectures. It specifically uses
// cuda::std::plus<> instead of a user-defined operator, which CUB recognizes to select an optimized code path.

// Tuning parameters found for signed integer types apply equally for unsigned integer types

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

// TODO(bgruber): let's add __half and __nv_bfloat16 eventually when they compile, since we have fast paths for them.
using value_types = all_types;
using op_t        = ::cuda::std::plus<>;
#include "base.cuh"
