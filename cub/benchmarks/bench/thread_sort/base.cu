// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/thread/thread_sort.cuh>

#include <cuda/std/functional>

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

enum class sort_algorithm
{
  stable,
  unstable
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  sort_algorithm,
  [](sort_algorithm value) {
    switch (value)
    {
      case sort_algorithm::stable:
        return "stable";
      case sort_algorithm::unstable:
        return "unstable";
    }
    return "unknown";
  },
  [](auto) {
    return std::string{};
  })

template <sort_algorithm Algorithm, int ItemsPerThread>
struct benchmark_op_t
{
  template <typename KeyT>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread]) const
  {
    cub::NullType values[ItemsPerThread];
    sort(keys, values);
    avoid_code_elimitation(keys);
  }

  template <typename KeyT, typename ValueT>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    sort(keys, values);
    avoid_code_elimitation(keys);
    avoid_code_elimitation(values);
  }

private:
  template <typename T>
  __device__ __forceinline__ static void avoid_code_elimitation(T (&values)[ItemsPerThread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      values[i] ^= static_cast<T>(i);
    }
  }

  template <typename KeyT, typename ValueT>
  __device__ __forceinline__ static void sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread])
  {
    if constexpr (Algorithm == sort_algorithm::stable)
    {
      cub::StableOddEvenSort(keys, values, cuda::std::less<>{});
    }
    else
    {
      cub::UnstablePairwiseSort(keys, values, cuda::std::less<>{});
    }
  }
};

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread>
void run_benchmark(nvbench::state& state)
{
  constexpr int block_size     = 256;
  constexpr int num_iterations = 128;
  using op_t                   = benchmark_op_t<Algorithm, ItemsPerThread>;
  const auto& kernel           = benchmark_kernel<ItemsPerThread, KeyT, ValueT, op_t>;

  const int num_sms = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_sm{};
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));
  const int grid_size = max_blocks_per_sm * num_sms;

  state.add_element_count(static_cast<size_t>(grid_size) * block_size * ItemsPerThread * num_iterations);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(num_iterations, op_t{});
  });
}

//----------------------------------------------------------------------------------------------------------------------
// parameter lists

using algorithms  = nvbench::enum_type_list<sort_algorithm::stable, sort_algorithm::unstable>;
using key_types   = nvbench::type_list<uint32_t, uint64_t>;
using value_types = nvbench::type_list<uint32_t, uint64_t>;
using item_counts = nvbench::enum_type_list<3, 4, 7, 8, 15, 16, 31, 32, 63, 64>;

// key-only sorting benchmarks
template <sort_algorithm Algorithm, typename KeyT, int ItemsPerThread>
void keys(nvbench::state& state,
          nvbench::type_list<nvbench::enum_type<Algorithm>, KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<Algorithm, KeyT, void, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(algorithms, key_types, item_counts))
  .set_type_axes_names({"algorithm", "KeyT", "items"});

// key-value sorting benchmarks
template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread>
void pairs(nvbench::state& state,
           nvbench::type_list<nvbench::enum_type<Algorithm>, KeyT, ValueT, nvbench::enum_type<ItemsPerThread>>)
{
  run_benchmark<Algorithm, KeyT, ValueT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(pairs, NVBENCH_TYPE_AXES(algorithms, key_types, value_types, item_counts))
  .set_type_axes_names({"algorithm", "KeyT", "ValueT", "items"});
