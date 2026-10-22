// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.cuh"

using item_counts = nvbench::enum_type_list<7, 8, 15, 16, 31, 32, 63, 64>;

template <typename KeyT, int ItemsPerThread>
void top_5(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::top_5, KeyT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(top_5, NVBENCH_TYPE_AXES(key_types, item_counts)).set_type_axes_names({"KeyT", "items"});
