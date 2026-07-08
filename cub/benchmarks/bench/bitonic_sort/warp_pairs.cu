// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_sort.cuh>

#include <nvbench_helper.cuh>

#include "bitonic_common.cuh"

using modes       = nvbench::enum_type_list<Mode::Latency, Mode::Throughput>;
using key_types   = nvbench::type_list<int16_t, float>;
using value_types = offset_types;
using len_values  = nvbench::enum_type_list<32, 64, 96, 128, 160, 192, 224, 256>;

template <int ITEMS_PER_THREAD>
struct full_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(keys, values, CustomLess{});
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void full(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<full_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ITEMS_PER_THREAD>
struct partial_oob_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int len) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(
      keys, values, CustomLess{}, len, CustomLess::oob_default<KeyT>);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void partial_oob(nvbench::state& state,
                 nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<partial_oob_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(partial_oob, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ITEMS_PER_THREAD>
struct partial_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int len) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(keys, values, CustomLess{}, len);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void partial(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<partial_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});
