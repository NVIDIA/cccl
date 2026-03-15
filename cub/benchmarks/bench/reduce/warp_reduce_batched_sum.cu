// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

using value_types = nvbench::type_list<int32_t, int64_t, float, double>;

using op_t = ::cuda::std::plus<>;

#include "warp_reduce_batched_base.cuh"
