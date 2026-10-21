// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 4:32:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 7:9:1
// %RANGE% TUNE_TRANSPOSE trp 0:1:1

using op_t = ::cuda::std::plus<>;

#include "variable_base.cuh"
