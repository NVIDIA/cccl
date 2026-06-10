// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// This benchmark uses a custom reduction operation, max_t, which is not known to CUB, so no operator specific
// optimizations (e.g. using redux or DPX instructions) are performed. This benchmark covers the unoptimized code path.

// Because CUB cannot detect this operator, we cannot add any tunings based on the results of this benchmark. Its main
// use is to detect regressions.

#include <nvbench_helper.cuh>

using value_types = all_types;
using op_t        = max_t;
#include "base.cuh"
