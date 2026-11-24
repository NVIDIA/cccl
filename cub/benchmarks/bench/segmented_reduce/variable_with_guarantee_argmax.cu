// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Note: This is not a tuning benchmark yet, since the max segmented size guarantee is not publicly exposed.

#include <nvbench_helper.cuh>

using op_t = cub::detail::arg_max;

#include "variable_with_guarantee_base.cuh"
