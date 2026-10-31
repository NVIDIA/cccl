// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_topk.cuh>

#include <cstdint>

#include <nvbench_helper.cuh>

#include "bitonic_common.cuh"

using modes        = nvbench::enum_type_list<BenchMode::Latency, BenchMode::Throughput>;
using key_types    = nvbench::type_list<std::int16_t, float>;
using len_values   = nvbench::enum_type_list<32, 64, 96, 128, 160>;
using max_k_values = nvbench::enum_type_list<32, 64, 96>;

// For best efficiency, max_k should be set to the smallest multiple of warp_threads that is not less than k.
// Choose k in the range (max_k - warp_threads, max_k] to reflect this.
[[nodiscard]] __device__ __host__ constexpr int calc_k(int max_k)
{
  return max_k - warp_threads / 3 * 2;
}

template <int WarpsPerBlock, int MaxK>
struct full_op_t
{
  template <typename KeyT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], int) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<KeyT, MaxK, warp_threads, cub::NullType>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];
    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(keys, CustomLess{}, calc_k(MaxK));
  }
};

template <BenchMode Mode, typename KeyT, int MaxK, int Len>
void full(nvbench::state& state,
          nvbench::type_list<nvbench::enum_type<Mode>, KeyT, nvbench::enum_type<MaxK>, nvbench::enum_type<Len>>)
{
  constexpr int warps_per_block = calc_run_params<Mode>().block_dim / warp_threads;
  run_topk_bench<full_op_t<warps_per_block, MaxK>, Mode, KeyT, void, Len, MaxK>(state, Len);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, key_types, max_k_values, len_values))
  .set_type_axes_names({"mode", "KeyT", "max_k", "len"});

// The partial overload with oob_default simply pads the input and calls the full overload, so it doesn't need a
// separate benchmark

template <int WarpsPerBlock, int MaxK>
struct partial_op_t
{
  template <typename KeyT, int ItemsPerThread>
  __device__ __forceinline__ void operator()(KeyT (&keys)[ItemsPerThread], int valid_items) const
  {
    using warp_topk_t = cub::detail::WarpBitonicTopK<KeyT, MaxK, warp_threads, cub::NullType>;
    __shared__ typename warp_topk_t::TempStorage temp_storage[WarpsPerBlock];

    warp_topk_t{temp_storage[threadIdx.x / warp_threads]}.TopK(keys, CustomLess{}, calc_k(MaxK), valid_items);
  }
};

template <BenchMode Mode, typename KeyT, int MaxK, int Len>
void partial(nvbench::state& state,
             nvbench::type_list<nvbench::enum_type<Mode>, KeyT, nvbench::enum_type<MaxK>, nvbench::enum_type<Len>>)
{
  constexpr int warps_per_block = calc_run_params<Mode>().block_dim / warp_threads;
  // Set valid_items to Len - 1 because:
  // (1) It reflects a realistic workload in which most items are valid. Because for best efficiency,
  //     ItemsPerThread should be ceil(valid_items / warp_threads).
  // (2) It exercises the partial-processing branch.
  // (3) It matches full_op_t's workload and avoids adding a valid_items axis that would bloat benchmark combinations.
  run_topk_bench<partial_op_t<warps_per_block, MaxK>, Mode, KeyT, void, Len, MaxK>(state, Len - 1);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, key_types, max_k_values, len_values))
  .set_type_axes_names({"mode", "KeyT", "max_k", "len"});
