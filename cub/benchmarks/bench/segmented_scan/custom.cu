// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This benchmark uses a custom operation, max_t, which is not known to CUB, so no operator specific optimizations and
// tunings are performed.

// Because CUB cannot detect this operator, we cannot add any tunings based on the results of this benchmark. Its main
// use is to detect regressions.

#include <nvbench_helper.cuh>

using op_t = max_t;
#include "base.cuh"
