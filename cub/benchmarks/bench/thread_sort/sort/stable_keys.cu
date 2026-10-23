// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.cuh"

using algorithms = nvbench::enum_type_list<sort_algorithm::stable>;

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(algorithms, key_types, item_counts))
  .set_type_axes_names({"algorithm", "KeyT", "items"});
