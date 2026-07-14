// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_sort.cuh>

#include <nvbench_helper.cuh>

#include "bitonic_common.cuh"

using modes       = nvbench::enum_type_list<Mode::Latency, Mode::Throughput>;
using key_types   = nvbench::type_list<int16_t, float>;
using value_types = offset_types;
using len_values  = nvbench::enum_type_list<32, 64, 96, 128, 160, 192, 224, 256>;

template <int ItemsPerThread>
struct full_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int) const
  {
    using WarpBitonicSort = cub::detail::WarpBitonicSort<ItemsPerThread, KeyT, ValueT>;
    using TempStorage     = typename WarpBitonicSort::TempStorage;
    __shared__ TempStorage temp_storage[32];
    const auto warp_id = threadIdx.x / 32;
    WarpBitonicSort{temp_storage[warp_id]}.Sort(keys, values, CustomLess{});
  }
};

template <Mode mode, typename KeyT, typename ValueT, int Len>
void full(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<Len>>)
{
  run_bench<full_op_t<Len / warp_threads>, mode, KeyT, ValueT, Len>(state);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ItemsPerThread>
struct partial_oob_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int len) const
  {
    using WarpBitonicSort = cub::detail::WarpBitonicSort<ItemsPerThread, KeyT, ValueT>;
    using TempStorage     = typename WarpBitonicSort::TempStorage;
    __shared__ TempStorage temp_storage[32];
    const auto warp_id = threadIdx.x / 32;
    WarpBitonicSort{temp_storage[warp_id]}.Sort(keys, values, CustomLess{}, len, CustomLess::oob_default<KeyT>);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int Len>
void partial_oob(nvbench::state& state,
                 nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<Len>>)
{
  run_bench<partial_oob_op_t<Len / warp_threads>, mode, KeyT, ValueT, Len>(state);
}

NVBENCH_BENCH_TYPES(partial_oob, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ItemsPerThread>
struct partial_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int len) const
  {
    using WarpBitonicSort = cub::detail::WarpBitonicSort<ItemsPerThread, KeyT, ValueT>;
    using TempStorage     = typename WarpBitonicSort::TempStorage;
    __shared__ TempStorage temp_storage[32];
    const auto warp_id = threadIdx.x / 32;
    WarpBitonicSort{temp_storage[warp_id]}.Sort(keys, values, CustomLess{}, len);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int Len>
void partial(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<Len>>)
{
  run_bench<partial_op_t<Len / warp_threads>, mode, KeyT, ValueT, Len>(state);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});
