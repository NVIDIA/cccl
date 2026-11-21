// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nvbench_helper.cuh>

#include "variable_common.cuh"

// No guarantee (max segment size of 0): every segment takes the large segment path
NVBENCH_BENCH_TYPES(variable_with_guarantee_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_default")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 16, 1))
  .add_int64_axis("GuaranteedMaxSegSize", {0});

// Small segments: 1-16 items per segment
NVBENCH_BENCH_TYPES(variable_with_guarantee_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_small_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 4, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(1, 4, 1));

// Medium segments: 32-256 items per segment
NVBENCH_BENCH_TYPES(variable_with_guarantee_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_medium_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(5, 8, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(5, 8, 1));

// Large segments: 512+ items per segment
NVBENCH_BENCH_TYPES(variable_with_guarantee_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_large_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(9, 16, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(9, 16, 1));
