// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

#include "bitonic_common.cuh"

using cub::detail::WarpBitonicTopKAlgorithm;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  WarpBitonicTopKAlgorithm,
  [](WarpBitonicTopKAlgorithm value) {
    switch (value)
    {
      case WarpBitonicTopKAlgorithm::eager:
        return "eager";
      case WarpBitonicTopKAlgorithm::buffered:
        return "buffered";
      default:
        return "Unknown";
    }
  },
  [](auto) {
    return std::string{};
  })

using modes        = nvbench::enum_type_list<Mode::Latency, Mode::Throughput>;
using algos        = nvbench::enum_type_list<WarpBitonicTopKAlgorithm::eager, WarpBitonicTopKAlgorithm::buffered>;
using eager_algos  = nvbench::enum_type_list<WarpBitonicTopKAlgorithm::eager>;
using key_types    = nvbench::type_list<std::int16_t, float>;
using value_types  = offset_types;
using len_values   = nvbench::enum_type_list<32, 64, 96, 128, 160>;
using max_k_values = nvbench::enum_type_list<32, 64, 96>;

// For array API:
// (1) k is set to 1 because performance is mostly determined by max_k not k
// (2) num_items is always set to len (ItemsPerThread * 32), though partial variants accept smaller values.
//     Because adding a num_items axis would bloat the benchmark combinations, and using a len much larger than
//     num_items is inefficient for these APIs.

template <WarpBitonicTopKAlgorithm Algo, int WarpsPerBlock, int MaxK>
struct full_op_t
{
  template <typename KeyT, typename ValueT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, ValueT, Algo>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(keys, values, CustomLess{}, 1);
  }
};

template <Mode BenchMode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int Len, int MaxK>
void full(nvbench::state& state,
          nvbench::type_list<nvbench::enum_type<BenchMode>,
                             nvbench::enum_type<Algo>,
                             KeyT,
                             ValueT,
                             nvbench::enum_type<Len>,
                             nvbench::enum_type<MaxK>>)
{
  constexpr int warps_per_block = get_run_params<BenchMode>().block_dim / warp_threads;
  run_topk<full_op_t<Algo, warps_per_block, MaxK>, BenchMode, KeyT, ValueT, Len, MaxK>(state);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, algos, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "len", "max_k"});

template <WarpBitonicTopKAlgorithm Algo, int WarpsPerBlock, int MaxK>
struct partial_oob_op_t
{
  template <typename KeyT, typename ValueT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, ValueT, Algo>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(
      keys, values, CustomLess{}, 1, ItemsPerThread * warp_threads, CustomLess::oob_default<KeyT>);
  }
};

template <Mode BenchMode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int Len, int MaxK>
void partial_oob(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<BenchMode>,
                     nvbench::enum_type<Algo>,
                     KeyT,
                     ValueT,
                     nvbench::enum_type<Len>,
                     nvbench::enum_type<MaxK>>)
{
  constexpr int warps_per_block = get_run_params<BenchMode>().block_dim / warp_threads;
  run_topk<partial_oob_op_t<Algo, warps_per_block, MaxK>, BenchMode, KeyT, ValueT, Len, MaxK>(state);
}

NVBENCH_BENCH_TYPES(partial_oob,
                    NVBENCH_TYPE_AXES(modes, eager_algos, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "len", "max_k"});

template <WarpBitonicTopKAlgorithm Algo, int WarpsPerBlock, int MaxK>
struct partial_op_t
{
  template <typename KeyT, typename ValueT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, ValueT, Algo>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(
      keys, values, CustomLess{}, 1, ItemsPerThread * warp_threads);
  }
};

template <Mode BenchMode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int Len, int MaxK>
void partial(nvbench::state& state,
             nvbench::type_list<nvbench::enum_type<BenchMode>,
                                nvbench::enum_type<Algo>,
                                KeyT,
                                ValueT,
                                nvbench::enum_type<Len>,
                                nvbench::enum_type<MaxK>>)
{
  constexpr int warps_per_block = get_run_params<BenchMode>().block_dim / warp_threads;
  run_topk<partial_op_t<Algo, warps_per_block, MaxK>, BenchMode, KeyT, ValueT, Len, MaxK>(state);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, algos, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "len", "max_k"});

template <WarpBitonicTopKAlgorithm Algo, int WarpsPerBlock, int MaxK, typename KeyT, typename ValueT>
__global__ void iterator_topk_kernel(int num_iterations, KeyT* keys_in, ValueT* values_in, int k, int num_items)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, ValueT, Algo>;

  __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];

  const std::size_t warp_id      = static_cast<std::size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / warp_threads;
  const std::size_t input_offset = warp_id * num_items * num_iterations;

  KeyT keys_out[MaxK / warp_threads];
  ValueT values_out[MaxK / warp_threads];

  warp_topk_t warp_topk{temp_storage[threadIdx.x / warp_threads]};
  for (int i = 0; i < num_iterations; ++i)
  {
    const std::size_t offset = input_offset + i * num_items;
    warp_topk.TopK(keys_in + offset, values_in + offset, CustomLess{}, k, num_items, keys_out, values_out);
    sink(keys_out);
    sink(values_out);
  }
}

template <Mode BenchMode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int MaxK>
void iterator(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<BenchMode>, nvbench::enum_type<Algo>, KeyT, ValueT, nvbench::enum_type<MaxK>>)
{
  const int num_items = static_cast<int>(state.get_int64("len"));
  const int k         = static_cast<int>(state.get_int64("k"));

  if (MaxK > num_items || k > MaxK || k > num_items)
  {
    state.skip("Skipping workload where max_k > len, k > max_k, or k > len.");
    return;
  }

  constexpr int warps_per_block = get_run_params<BenchMode>().block_dim / warp_threads;
  const auto kernel             = iterator_topk_kernel<Algo, warps_per_block, MaxK, KeyT, ValueT>;

  RunParams run_params = get_run_params<BenchMode>();
  if (BenchMode == Mode::Throughput)
  {
    // scale grid_dim because throughout mode is slow
    run_params.grid_dim /= num_items / warp_threads;
  }
  const std::size_t input_items = count_items(run_params, num_items);

  thrust::device_vector<KeyT> keys_in     = generate(input_items);
  thrust::device_vector<ValueT> values_in = generate(input_items);

  state.add_element_count(input_items);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<run_params.grid_dim, run_params.block_dim, 0, launch.get_stream()>>>(
      run_params.num_iterations,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(values_in.data()),
      k,
      num_items);
  });
}

NVBENCH_BENCH_TYPES(iterator, NVBENCH_TYPE_AXES(modes, algos, key_types, value_types, max_k_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "max_k"})
  .add_int64_axis("len", {32, 64, 96, 128, 256, 512, 1024, 2048})
  .add_int64_axis("k", {32});
