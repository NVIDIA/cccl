// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_radix_sort.cuh>

#include <nvbench_helper.cuh>

// %//RANGE//% TUNE_RADIX_BITS bits 8:9:1
#define TUNE_RADIX_BITS 8

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#include "policy_selector.h"

template <typename KeyT, typename ValueT, typename OffsetT>
void radix_sort_values(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  if constexpr (!fits_in_default_shared_memory<KeyT, ValueT, OffsetT, cub::SortOrder::Ascending>())
  {
    return;
  }

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<KeyT> keys_in = generate(elements, entropy);
  thrust::device_vector<KeyT> keys_out(elements, thrust::no_init);
  thrust::device_vector<ValueT> values_in = generate(elements);
  thrust::device_vector<ValueT> values_out(elements, thrust::no_init);

  const KeyT* d_keys_in     = thrust::raw_pointer_cast(keys_in.data());
  KeyT* d_keys_out          = thrust::raw_pointer_cast(keys_out.data());
  const ValueT* d_values_in = thrust::raw_pointer_cast(values_in.data());
  ValueT* d_values_out      = thrust::raw_pointer_cast(values_out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<KeyT, ValueT, OffsetT>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceRadixSort::SortPairs,
      "SortPairs failed",
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      static_cast<OffsetT>(elements),
      0,
      sizeof(KeyT) * 8,
      env);
  });
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = integral_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(Tune_ValueT)
using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#  if _CCCL_HAS_INT128()
                     ,
                     int128_t
#  endif
                     >;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(radix_sort_values, NVBENCH_TYPE_AXES(key_types, value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
