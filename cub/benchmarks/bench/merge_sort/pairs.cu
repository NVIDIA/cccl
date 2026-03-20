// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_merge_sort.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#ifndef TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
#endif

#if !TUNE_BASE
template <typename KeyT>
struct policy_selector
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> cub::detail::merge_sort::merge_sort_policy
  {
    return cub::detail::merge_sort::merge_sort_policy{
      TUNE_THREADS_PER_BLOCK,
      cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
      (TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE),
      (TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : (TUNE_LOAD == 1 ? cub::LOAD_LDG : cub::LOAD_CA)),
      (TUNE_TRANSPOSE == 0 ? cub::BLOCK_STORE_DIRECT : cub::BLOCK_STORE_WARP_TRANSPOSE)};
  }
};
#endif // TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
void pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using key_t        = KeyT;
  using value_t      = ValueT;
  using offset_t     = cub::detail::choose_offset_t<OffsetT>;
  using compare_op_t = less_t;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<key_t> keys_buffer_1 = generate(elements, entropy);
  thrust::device_vector<key_t> keys_buffer_2(elements);
  thrust::device_vector<value_t> values_buffer_1(elements);
  thrust::device_vector<value_t> values_buffer_2(elements);

  key_t* d_keys_buffer_1     = thrust::raw_pointer_cast(keys_buffer_1.data());
  key_t* d_keys_buffer_2     = thrust::raw_pointer_cast(keys_buffer_2.data());
  value_t* d_values_buffer_1 = thrust::raw_pointer_cast(values_buffer_1.data());
  value_t* d_values_buffer_2 = thrust::raw_pointer_cast(values_buffer_2.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  cub::detail::merge_sort::dispatch(
    nullptr,
    temp_size,
    d_keys_buffer_1,
    d_values_buffer_1,
    d_keys_buffer_2,
    d_values_buffer_2,
    static_cast<offset_t>(elements),
    compare_op_t{},
    0 /* stream */
#if !TUNE_BASE
    ,
    policy_selector<key_t>{}
#endif // !TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::merge_sort::dispatch(
      temp_storage,
      temp_size,
      d_keys_buffer_1,
      d_values_buffer_1,
      d_keys_buffer_2,
      d_values_buffer_2,
      static_cast<offset_t>(elements),
      compare_op_t{},
      launch.get_stream()
#if !TUNE_BASE
        ,
      policy_selector<key_t>{}
#endif // !TUNE_BASE
    );
  });
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = all_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t
#  if _CCCL_HAS_INT128()
// nvcc currently hangs for __int128 value type with the fallback policy of {CTA: 64, IPT: 1}. NVBug 4384075
//  ,
//  int128_t
#  endif
                                       >;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(pairs, NVBENCH_TYPE_AXES(key_types, value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
