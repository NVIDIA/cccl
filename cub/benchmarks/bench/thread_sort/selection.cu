// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/thread/thread_sort.cuh>

#include <cuda/ptx>
#include <cuda/std/functional>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

enum class result_kind
{
  median = 1,
  top_3  = 3,
  top_5  = 5,
  top_16 = 16,
};

template <int ItemsPerThread>
struct benchmark_op_t
{
  template <typename KeyT>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread]) const
  {
    cub::NullType values[ItemsPerThread];
    cub::UnstablePairwiseSort(keys, values, cuda::std::less<>{});
  }
};

template <int BlockSize, result_kind Result, int ItemsPerThread, typename KeyT, typename ActionT>
__global__ void selection_benchmark_kernel(int num_iterations, ActionT action)
{
  constexpr int warp_threads = 32;
  __shared__ volatile KeyT output[BlockSize];
  KeyT keys[ItemsPerThread];
  const auto tid = threadIdx.x;
  uint32_t seed  = cuda::ptx::get_sreg_clock() + (tid + 7) % warp_threads;

#pragma unroll 1
  for (int iter = 0; iter < num_iterations; ++iter)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item] = generate_random_data<KeyT>(seed);
    }
    action(keys);

    if constexpr (Result == result_kind::median)
    {
      output[tid] = keys[ItemsPerThread / 2];
    }
    else
    {
      constexpr int output_count = static_cast<int>(Result);
      static_assert(output_count <= ItemsPerThread);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < output_count; ++item)
      {
        output[tid] = keys[item];
      }
    }
  }
}

template <result_kind Result, typename KeyT, int ItemsPerThread>
void run_benchmark(nvbench::state& state)
{
  constexpr int block_size     = 256;
  constexpr int num_iterations = 128;
  using op_t                   = benchmark_op_t<ItemsPerThread>;
  const auto& kernel           = selection_benchmark_kernel<block_size, Result, ItemsPerThread, KeyT, op_t>;

  const int num_sms = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_sm{};
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));
  const int grid_size = max_blocks_per_sm * num_sms;

  state.add_element_count(static_cast<size_t>(grid_size) * block_size * ItemsPerThread * num_iterations);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream().get_stream()>>>(num_iterations, op_t{});
  });
}

using key_types          = nvbench::type_list<uint32_t, uint64_t>;
using item_counts        = nvbench::enum_type_list<4, 7, 8, 15, 16, 31, 32, 63, 64>;
using top_5_item_counts  = nvbench::enum_type_list<7, 8, 15, 16, 31, 32, 63, 64>;
using top_16_item_counts = nvbench::enum_type_list<16, 31, 32, 63, 64>;

template <typename KeyT, int ItemsPerThread>
void median(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::median, KeyT, ItemsPerThread>(state);
}

template <typename KeyT, int ItemsPerThread>
void top_3(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::top_3, KeyT, ItemsPerThread>(state);
}

template <typename KeyT, int ItemsPerThread>
void top_5(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::top_5, KeyT, ItemsPerThread>(state);
}

template <typename KeyT, int ItemsPerThread>
void top_16(nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<result_kind::top_16, KeyT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(median, NVBENCH_TYPE_AXES(key_types, item_counts)).set_type_axes_names({"KeyT", "items"});
NVBENCH_BENCH_TYPES(top_3, NVBENCH_TYPE_AXES(key_types, item_counts)).set_type_axes_names({"KeyT", "items"});
NVBENCH_BENCH_TYPES(top_5, NVBENCH_TYPE_AXES(key_types, top_5_item_counts)).set_type_axes_names({"KeyT", "items"});
NVBENCH_BENCH_TYPES(top_16, NVBENCH_TYPE_AXES(key_types, top_16_item_counts)).set_type_axes_names({"KeyT", "items"});
