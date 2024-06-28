// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "nvbench_helper.cuh"

template <typename T>
static void benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  const auto& name    = state.get_benchmark().get_name();
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> in(elements, T{1});
  thrust::device_vector<T> b(elements, T{1});
  caching_allocator_t alloc;

  if (name == "first")
  {
    b.front() = T{2};
  }
  else if (name == "last")
  {
    b.back() = T{2};
  }
  else
  {
    const auto similarity     = state.get_float64("SimilarityRatio");
    const auto differentElems = std::min(static_cast<std::size_t>(elements * similarity), elements);
    thrust::fill(policy(alloc), b.begin() + differentElems, b.end(), T{2});
  }

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements);

  auto body = [&](auto&... args) {
    do_not_optimize(thrust::equal(policy(alloc, args...), in.begin(), in.end(), b.begin()));
  };
  body(); // discarded warmup run TODO(bgruber): is that actually needed?

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    body(launch);
  });
}

NVBENCH_BENCH_TYPES(benchmark, NVBENCH_TYPE_AXES(integral_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("SimilarityRatio", std::vector{1.0, 0.5, 0.0});

NVBENCH_BENCH_TYPES(benchmark, NVBENCH_TYPE_AXES(integral_types))
  .set_name("first")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

NVBENCH_BENCH_TYPES(benchmark, NVBENCH_TYPE_AXES(integral_types))
  .set_name("last")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
