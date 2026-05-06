// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>

#include <cuda/functional>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  using select_op_t = less_then_t<T>;

  // set up input
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = ::cuda::std::clamp<std::size_t>(elements * common_prefix, 0ull, elements - 1);

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  thrust::sequence(dinput.begin(), dinput.end(), T{0});

  state.add_global_memory_reads<T>(2 * elements);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(thrust::is_partitioned(
                 policy(alloc, launch), dinput.begin(), dinput.end(), select_op_t{static_cast<T>(mismatch_point)}));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});
