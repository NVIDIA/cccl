// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

template <int BlockSize>
struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using BlockScan   = cub::BlockScan<T, BlockSize>;
    using TempStorage = typename BlockScan::TempStorage;
    __shared__ TempStorage temp_storage;
    T inclusive_output;
    BlockScan{temp_storage}.InclusiveScan(thread_data, inclusive_output, op_t{});
    // BlockScan::TempStorage is real shared memory, unlike WarpScanShfl's empty type, so the
    // chained calls in benchmark_kernel all reuse one instance of it. BlockScan documents that
    // "a subsequent __syncthreads() threadblock barrier should be invoked after calling this
    // method if the collective's temporary storage is to be reused or repurposed", so the
    // barrier is required here for correctness, not as a precaution. It therefore falls inside
    // the timed region, which is the honest place for it: any caller that reuses one
    // TempStorage in a loop pays the same barrier every iteration.
    __syncthreads();
    return inclusive_output;
  }
};

template <typename T>
void block_scan(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_size    = 256;
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, benchmark_op_t<block_size>, T>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  // NVBENCH_CUDA_CALL_NOEXCEPT swallows a failed occupancy query, which would leave
  // max_blocks_per_SM at 0 and turn the launch into a bare cudaErrorInvalidConfiguration.
  if (max_blocks_per_SM == 0)
  {
    state.skip("Skipping: no resident blocks for this type on this device.");
    return;
  }
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(benchmark_op_t<block_size>{});
  });
}

NVBENCH_BENCH_TYPES(block_scan, NVBENCH_TYPE_AXES(value_types)).set_name("base").set_type_axes_names({"T{ct}"});
