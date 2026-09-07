// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This tunes the vectorized implementation of cub::DeviceTransform. It has distinct tuning parameters and is thus in a
// separate file. Use fill.cu for benchmarking.

#define TUNE_ALGORITHM 1

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_UNROLL_FACTOR unrl 1:4:1
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1

#include "fill.h"
