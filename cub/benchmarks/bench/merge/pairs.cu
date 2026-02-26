// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_merge.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/utility>

#include <cstdint>

#include <nvbench_helper.cuh>

#include "merge_common.cuh"

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:3:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#if !TUNE_BASE
struct bench_policy_selector
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> cub::detail::merge::merge_policy
  {
    return cub::detail::merge::merge_policy{
      TUNE_THREADS_PER_BLOCK,
      cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
      TUNE_LOAD_MODIFIER,
      TUNE_STORE_ALGORITHM,
      TUNE_USE_BL2SH};
  }
};
#endif // !TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
void pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using compare_op_t = less_t;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const auto num_items_lhs = elements / 2;
  const auto num_items_rhs = elements - num_items_lhs;

  thrust::device_vector<KeyT> keys_out(elements);
  thrust::device_vector<ValueT> values_lhs(num_items_lhs);
  thrust::device_vector<ValueT> values_rhs(num_items_rhs);
  thrust::device_vector<ValueT> values_out(elements);

  auto [keys_lhs, keys_rhs] = generate_lhs_rhs<KeyT>(num_items_lhs, num_items_rhs, entropy);

  KeyT* d_keys_lhs     = thrust::raw_pointer_cast(keys_lhs.data());
  KeyT* d_keys_rhs     = thrust::raw_pointer_cast(keys_rhs.data());
  KeyT* d_keys_out     = thrust::raw_pointer_cast(keys_out.data());
  ValueT* d_values_lhs = thrust::raw_pointer_cast(values_lhs.data());
  ValueT* d_values_rhs = thrust::raw_pointer_cast(values_rhs.data());
  ValueT* d_values_out = thrust::raw_pointer_cast(values_out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  cub::detail::merge::dispatch(
    nullptr,
    temp_size,
    d_keys_lhs,
    d_values_lhs,
    static_cast<OffsetT>(num_items_lhs),
    d_keys_rhs,
    d_values_rhs,
    static_cast<OffsetT>(num_items_rhs),
    d_keys_out,
    d_values_out,
    compare_op_t{},
    cudaStream_t{}
#if !TUNE_BASE
    ,
    bench_policy_selector{}
#endif // !TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::merge::dispatch(
      temp_storage,
      temp_size,
      d_keys_lhs,
      d_values_lhs,
      static_cast<OffsetT>(num_items_lhs),
      d_keys_rhs,
      d_values_rhs,
      static_cast<OffsetT>(num_items_rhs),
      d_keys_out,
      d_values_out,
      compare_op_t{},
      launch.get_stream()
#if !TUNE_BASE
        ,
      bench_policy_selector{}
#endif // !TUNE_BASE
    );
  });
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = fundamental_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t
#  if NVBENCH_HELPER_HAS_I128
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
