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
using key_types    = nvbench::type_list<std::int16_t, float>;
using value_types  = nvbench::type_list<int32_t>;
using len_values   = nvbench::enum_type_list<32, 64, 96, 128, 160>;
using max_k_values = nvbench::enum_type_list<32, 64, 96>;

// For best efficiency, max_k should be set to the smallest multiple of warp_threads that is not less than k.
// Choose k in the range (max_k - warp_threads, max_k] to reflect this.
[[nodiscard]] __device__ __host__ constexpr int calc_k(int max_k)
{
  return max_k - warp_threads / 3 * 2;
}

template <int WarpsPerBlock, int MaxK, WarpBitonicTopKAlgorithm Algo>
struct full_op_t
{
  template <typename KeyT, typename ValueT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, warp_threads, ValueT, Algo>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(keys, values, CustomLess{}, calc_k(MaxK));
  }
};

template <Mode mode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int MaxK, int Len>
void full(nvbench::state& state,
          nvbench::type_list<nvbench::enum_type<mode>,
                             nvbench::enum_type<Algo>,
                             KeyT,
                             ValueT,
                             nvbench::enum_type<MaxK>,
                             nvbench::enum_type<Len>>)
{
  constexpr int warps_per_block = calc_run_params<mode>().block_dim / warp_threads;
  run_topk_bench<full_op_t<warps_per_block, MaxK, Algo>, mode, KeyT, ValueT, Len, MaxK>(state);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, algos, key_types, value_types, max_k_values, len_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "max_k", "len"});

// The partial overload with oob_default performs nearly the same as the full overload, so do not benchmark it

template <int WarpsPerBlock, int MaxK, WarpBitonicTopKAlgorithm Algo>
struct partial_op_t
{
  template <typename KeyT, typename ValueT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread]) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, warp_threads, ValueT, Algo>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];

    // All items are assumed valid, because for best efficiency, ItemsPerThread should be set to
    // ceil(valid_nums / warp_threads). This also matches full_op_t's workload and avoids adding a
    // valid_items axis that would bloat benchmark combinations.
    constexpr int valid_nums = ItemsPerThread * warp_threads;
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(keys, values, CustomLess{}, calc_k(MaxK), valid_nums);
  }
};

template <Mode mode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int MaxK, int Len>
void partial(nvbench::state& state,
             nvbench::type_list<nvbench::enum_type<mode>,
                                nvbench::enum_type<Algo>,
                                KeyT,
                                ValueT,
                                nvbench::enum_type<MaxK>,
                                nvbench::enum_type<Len>>)
{
  constexpr int warps_per_block = calc_run_params<mode>().block_dim / warp_threads;
  run_topk_bench<partial_op_t<warps_per_block, MaxK, Algo>, mode, KeyT, ValueT, Len, MaxK>(state);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, algos, key_types, value_types, max_k_values, len_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "max_k", "len"});

template <int WarpsPerBlock, int MaxK, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT>
__global__ void iterator_kernel(int num_iterations, KeyT* keys_in, ValueT* values_in, int k, int num_items)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, warp_threads, ValueT, Algo>;

  __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];

  const std::size_t warp_id      = static_cast<std::size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / warp_threads;
  const std::size_t input_offset = warp_id * num_items * num_iterations;

  KeyT keys_out[MaxK / warp_threads];
  ValueT values_out[MaxK / warp_threads];

  warp_topk_t warp_topk{temp_storage[threadIdx.x / warp_threads]};
  for (int i = 0; i < num_iterations; ++i)
  {
    const std::size_t offset            = input_offset + i * num_items;
    constexpr int load_items_per_thread = 2;
    warp_topk.TopK<load_items_per_thread>(
      keys_in + offset, values_in + offset, CustomLess{}, k, num_items, keys_out, values_out);
    sink(keys_out);
    sink(values_out);
  }
}

template <Mode mode, WarpBitonicTopKAlgorithm Algo, typename KeyT, typename ValueT, int MaxK>
void iterator(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<mode>, nvbench::enum_type<Algo>, KeyT, ValueT, nvbench::enum_type<MaxK>>)
{
  const int num_items = static_cast<int>(state.get_int64("len"));
  constexpr int k     = calc_k(MaxK);

  if (MaxK > num_items || k > num_items)
  {
    state.skip("Skipping workload where max_k > len or k > len.");
    return;
  }

  constexpr int warps_per_block = calc_run_params<mode>().block_dim / warp_threads;
  const auto kernel             = iterator_kernel<warps_per_block, MaxK, Algo, KeyT, ValueT>;

  const RunParams run_params    = calc_topk_run_params<mode>(num_items);
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

NVBENCH_BENCH_TYPES(
  iterator,
  NVBENCH_TYPE_AXES(
    modes, nvbench::enum_type_list<WarpBitonicTopKAlgorithm::buffered>, key_types, value_types, max_k_values))
  .set_type_axes_names({"mode", "algo", "KeyT", "ValueT", "max_k"})
  .add_int64_axis("len", {32, 64, 96, 128, 256, 512, 1024, 2048});
