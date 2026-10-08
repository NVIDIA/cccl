// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

// complex types cannot be compared with operator<
using value_types = nvbench::type_list<bool>;

using op_t = ::cuda::std::logical_and<>;
#include "warp_reduce_base.cuh"
