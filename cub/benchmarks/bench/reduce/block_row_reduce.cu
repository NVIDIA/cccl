// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_row_reduce.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

template <int RowsPerBlock, int WarpsPerRow>
struct shared_broadcast_t
{
  template <typename T>
  static constexpr int block_threads = cub::BlockRowReduce<T, RowsPerBlock, WarpsPerRow>::BLOCK_THREADS;

  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using row_reduce_t = cub::BlockRowReduce<T, RowsPerBlock, WarpsPerRow>;

    __shared__ typename row_reduce_t::TempStorage temp_storage;

    const T result = row_reduce_t{temp_storage}.Sum(thread_data);
    if constexpr (WarpsPerRow > 1)
    {
      __syncthreads();
    }
    return result;
  }
};

template <int RowsPerBlock, int WarpsPerRow>
struct warp_broadcast_t
{
  template <typename T>
  static constexpr int block_threads = cub::BlockRowReduceWarpBroadcast<T, RowsPerBlock, WarpsPerRow>::BLOCK_THREADS;

  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using row_reduce_t = cub::BlockRowReduceWarpBroadcast<T, RowsPerBlock, WarpsPerRow>;

    __shared__ typename row_reduce_t::TempStorage temp_storage;

    const T result = row_reduce_t{temp_storage}.Sum(thread_data);
    if constexpr (WarpsPerRow > 1)
    {
      __syncthreads();
    }
    return result;
  }
};

using row_reduce_variants = nvbench::type_list<
  shared_broadcast_t<1, 1>,
  warp_broadcast_t<1, 1>,
  shared_broadcast_t<1, 4>,
  warp_broadcast_t<1, 4>,
  shared_broadcast_t<2, 4>,
  warp_broadcast_t<2, 4>,
  shared_broadcast_t<4, 4>,
  warp_broadcast_t<4, 4>,
  shared_broadcast_t<1, 16>,
  warp_broadcast_t<1, 16>,
  shared_broadcast_t<1, 32>,
  warp_broadcast_t<1, 32>>;
using value_types = nvbench::type_list<int, float, double>;

template <typename RowReduceVariant, typename T>
void block_row_reduce(nvbench::state& state, nvbench::type_list<RowReduceVariant, T>)
{
  constexpr int block_size = RowReduceVariant::template block_threads<T>;
  // device_side_benchmark.cuh initializes data from registers and sinks through an unreachable global write, so the
  // measured loop intentionally isolates the unrolled primitive body from ordinary global memory traffic.
  constexpr int unroll_factor = 64; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, RowReduceVariant, T>;
  const int num_sms           = state.get_device().value().get_number_of_sms();
  int max_blocks_per_sm       = 0;

  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));
  if (max_blocks_per_sm <= 0)
  {
    state.skip("Skipping: benchmark kernel does not fit on the selected device.");
    return;
  }

  const int grid_size = max_blocks_per_sm * num_sms;
  state.add_element_count(grid_size * block_size * unroll_factor, "Thread reductions");
  state.add_global_memory_reads<T>(0);
  state.add_global_memory_writes<T>(0);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(RowReduceVariant{});
  });
}

NVBENCH_BENCH_TYPES(block_row_reduce, NVBENCH_TYPE_AXES(row_reduce_variants, value_types))
  .set_name("block_row_reduce")
  .set_type_axes_names({"Variant{ct}", "T{ct}"});
