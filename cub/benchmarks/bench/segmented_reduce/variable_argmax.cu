// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

using value_types       = nvbench::type_list<int32_t, int64_t, float, double>;
using op_t              = cub::detail::arg_max;
using some_offset_types = nvbench::type_list<int32_t>;

#include "variable_base.cuh"
