// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using WarpReduce  = cub::WarpReduce<T>;
    using TempStorage = typename WarpReduce::TempStorage;
    __shared__ TempStorage temp_storage[32];
    auto warp_id = threadIdx.x / 32;
    return WarpReduce{temp_storage[warp_id]}.Reduce(thread_data, op_t{});
  }
};

template <typename T>
void warp_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_size    = 256;
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  constexpr int total_items   = 1 << 28; // large enough to hide tail effects on current devices
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, benchmark_op_t, T>;
  const int grid_size         = total_items / (block_size * unroll_factor);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(benchmark_op_t{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce, NVBENCH_TYPE_AXES(value_types)).set_name("base").set_type_axes_names({"T{ct}"});

template <typename T>
void warp_reduce_latency(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_size    = 32; // single warp to measure latency
  constexpr int unroll_factor = 128;
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, benchmark_op_t, T>;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<1, block_size>>>(benchmark_op_t{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_latency, NVBENCH_TYPE_AXES(value_types))
  .set_name("latency")
  .set_type_axes_names({"T{ct}"});
