// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "variable_common.cuh"

NVBENCH_BENCH_TYPES(variable_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_default")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 16, 1));
