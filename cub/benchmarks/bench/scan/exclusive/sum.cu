// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Tuning parameters found for signed integer types apply equally for unsigned integer types

#include <nvbench_helper.cuh>

// This benchmark tunes the old, non-warpspeed scan implementation. Using it for benchmarking, will pick the warpspeed
// implementation on SM100+, but it's better to use the sum.warpspeed.cu benchmark instead, which uses a single OffsetT.

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#define USES_WARPSPEED() 0
using op_t              = ::cuda::std::plus<>;
using scan_offset_types = offset_types;
#include "base.cuh"
