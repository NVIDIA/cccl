// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.cuh"

using item_counts = nvbench::enum_type_list<16, 31, 32, 63, 64>;

template <typename KeyT, int ItemsPerThread>
void top_16(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::top_16, KeyT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(top_16, NVBENCH_TYPE_AXES(key_types, item_counts)).set_type_axes_names({"KeyT", "items"});
