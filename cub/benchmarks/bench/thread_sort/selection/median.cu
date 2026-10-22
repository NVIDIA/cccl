// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.cuh"

using item_counts = nvbench::enum_type_list<4, 7, 8, 15, 16>;

template <typename KeyT, int ItemsPerThread>
void median(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::median, KeyT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(median, NVBENCH_TYPE_AXES(key_types, item_counts)).set_type_axes_names({"KeyT", "items"});
