// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// This benchmark uses a custom operation, max_t, which is not known to CUB, so no operator specific optimizations and
// tunings are performed.

// Because CUB cannot detect this operator, we cannot add any tunings based on the results of this benchmark. Its main
// use is to detect regressions.

#include <nvbench_helper.cuh>

#define USES_WARPSPEED() 0
using op_t              = max_t;
using scan_offset_types = offset_types;
#include "base.cuh"
