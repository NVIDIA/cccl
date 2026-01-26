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
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  constexpr cub::SortOrder sort_order = cub::SortOrder::Ascending;
  constexpr bool is_overwrite_ok      = false;
  using key_t                         = KeyT;
  using value_t                       = ValueT;

  if constexpr (!fits_in_default_shared_memory<key_t, value_t, offset_t, sort_order>())
  {
    return;
  }

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(key_t) * 8;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<key_t> keys_buffer_1     = generate(elements, entropy);
  thrust::device_vector<value_t> values_buffer_1 = generate(elements);
  thrust::device_vector<key_t> keys_buffer_2(elements);
  thrust::device_vector<value_t> values_buffer_2(elements);

  key_t* d_keys_buffer_1     = thrust::raw_pointer_cast(keys_buffer_1.data());
  key_t* d_keys_buffer_2     = thrust::raw_pointer_cast(keys_buffer_2.data());
  value_t* d_values_buffer_1 = thrust::raw_pointer_cast(values_buffer_1.data());
  value_t* d_values_buffer_2 = thrust::raw_pointer_cast(values_buffer_2.data());

  cub::DoubleBuffer<key_t> d_keys(d_keys_buffer_1, d_keys_buffer_2);
  cub::DoubleBuffer<value_t> d_values(d_values_buffer_1, d_values_buffer_2);

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  cub::detail::radix_sort::dispatch<sort_order>(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    static_cast<offset_t>(elements),
    begin_bit,
    end_bit,
    is_overwrite_ok,
    0 /* stream */
#if !TUNE_BASE
    ,
    cub::detail::identity_decomposer_t{},
    policy_selector<KeyT>{}
#endif // !TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;

    cub::detail::radix_sort::dispatch<sort_order>(
      temp_storage,
      temp_size,
      keys,
      values,
      static_cast<offset_t>(elements),
      begin_bit,
      end_bit,
      is_overwrite_ok,
      launch.get_stream()
#if !TUNE_BASE
        ,
      cub::detail::identity_decomposer_t{},
      policy_selector<KeyT>{}
#endif // !TUNE_BASE
    );
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
#  if NVBENCH_HELPER_HAS_I128
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
