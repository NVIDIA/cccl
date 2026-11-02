// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/warp/warp_scan.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using WarpScan    = cub::WarpScan<T>;
    using TempStorage = typename WarpScan::TempStorage;
    __shared__ TempStorage temp_storage[32];
    auto warp_id = threadIdx.x / 32;
    T inclusive_output;
    WarpScan{temp_storage[warp_id]}.InclusiveScan(thread_data, inclusive_output, op_t{});
    return inclusive_output;
  }
};

template <typename T>
void warp_scan(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_size    = 256;
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, benchmark_op_t, T>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(benchmark_op_t{});
  });
}

NVBENCH_BENCH_TYPES(warp_scan, NVBENCH_TYPE_AXES(value_types)).set_name("base").set_type_axes_names({"T{ct}"});
