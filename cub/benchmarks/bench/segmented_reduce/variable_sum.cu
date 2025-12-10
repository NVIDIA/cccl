// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

using value_types       = nvbench::type_list<int32_t, int64_t, float, double>;
using op_t              = ::cuda::std::plus<>;
using some_offset_types = nvbench::type_list<int32_t>;

#include "variable_base.cuh"
