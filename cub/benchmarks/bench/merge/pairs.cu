// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/tabulate_output_iterator.h>

#include <cuda/std/utility>

#include <cstdint>

#include "thrust/detail/raw_pointer_cast.h"
#include <nvbench_helper.cuh>

template <typename T>
struct to_key_t
{
  template <typename IndexT>
  __host__ __device__ __forceinline__ T operator()(IndexT index) const
  {
    return static_cast<T>(index);
  }
};

struct select_if_less_than_t
{
  bool negate;
  uint8_t threshold;

  __host__ __device__ __forceinline__ bool operator()(uint8_t val) const
  {
    return negate ? !(val < threshold) : val < threshold;
  }
};

template <typename OffsetT>
struct write_pivot_point_t
{
  OffsetT threshold;
  OffsetT* pivot_point;

  __host__ __device__ __forceinline__ void operator()(OffsetT output_index, OffsetT input_index) const
  {
    if (output_index == threshold)
    {
      *pivot_point = input_index;
    }
  }
};

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#ifndef TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
#endif

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename KeyT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using merge_policy =
      cub::agent_policy_t<TUNE_THREADS_PER_BLOCK,
                          cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
                          TUNE_LOAD_ALGORITHM,
                          TUNE_LOAD_MODIFIER,
                          TUNE_STORE_ALGORITHM>;
  };

  using MaxPolicy = policy_t;
};
#endif // TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
void pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using key_t            = KeyT;
  using value_t          = ValueT;
  using key_input_it_t   = key_t*;
  using value_input_it_t = value_t*;
  using key_it_t         = key_t*;
  using value_it_t       = value_t*;
  using offset_t         = OffsetT;
  using compare_op_t     = less_t;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t>;
  using dispatch_t = cub::
    DispatchMergeSort<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::detail::merge::
    dispatch_t<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  // We generate data distributions in the range [0, 255] that, with lower entropy, get skewed towards 0
  // We use this to generate increasingly large *consecutive* segments of data that are getting selected from the lhs
  thrust::device_vector<uint8_t> rnd_selector_val = generate(elements, entropy);
  uint8_t threshold                               = 128;
  select_if_less_than_t select_lhs_op{false, threshold};
  select_if_less_than_t select_rhs_op{true, threshold};

  // The following algorithm only works under the precondition that there's at least 50% of the data in the lhs
  // If that's not the case, we simply swap the logic for selecting into lhs and rhs
  const auto num_items_selected_into_lhs =
    static_cast<offset_t>(thrust::count_if(rnd_selector_val.begin(), rnd_selector_val.end(), select_lhs_op));
  if (num_items_selected_into_lhs < elements / 2)
  {
    using ::cuda::std::swap;
    swap(select_lhs_op, select_rhs_op);
  }

  // We want lhs and rhs to be of equal size. We also want to have skewed distributions, such that we put different
  // workloads on the binary search part. For this reason, we identify the index from the input, referred to as pivot
  // point, after which the lhs is "full". We compose the rhs by selecting all unselected items up to the pivot point
  // and *all* items after the pivot point.
  constexpr std::size_t num_pivot_points = 1;
  thrust::device_vector<offset_t> pivot_point(num_pivot_points);
  const auto num_items_lhs = elements / 2;
  const auto num_items_rhs = elements - num_items_lhs;
  auto counting_it         = thrust::make_counting_iterator(offset_t{0});
  thrust::copy_if(
    counting_it,
    counting_it + elements,
    rnd_selector_val.begin(),
    thrust::make_tabulate_output_iterator(write_pivot_point_t<offset_t>{
      static_cast<offset_t>(num_items_lhs), thrust::raw_pointer_cast(pivot_point.data())}),
    select_lhs_op);

  thrust::device_vector<key_t> keys_lhs(num_items_lhs);
  thrust::device_vector<key_t> keys_rhs(num_items_rhs);
  thrust::device_vector<key_t> keys_out(elements);
  thrust::device_vector<value_t> values_lhs(num_items_lhs);
  thrust::device_vector<value_t> values_rhs(num_items_rhs);
  thrust::device_vector<value_t> values_out(elements);

  // Fancy iterator to generate key_t in strictly increasing order
  auto data_gen_it = thrust::make_transform_iterator(counting_it, to_key_t<key_t>{});

  // Select lhs from input up to pivot point
  offset_t pivot_point_val = pivot_point[0];
  auto const end_lhs       = thrust::copy_if(
    data_gen_it, data_gen_it + pivot_point_val, rnd_selector_val.cbegin(), keys_lhs.begin(), select_lhs_op);
  // Select rhs items from input up to pivot point
  auto const end_rhs = thrust::copy_if(
    data_gen_it, data_gen_it + pivot_point_val, rnd_selector_val.cbegin(), keys_rhs.begin(), select_rhs_op);
  // From pivot point copy all remaining items to rhs
  thrust::copy(data_gen_it + pivot_point_val, data_gen_it + elements, end_rhs);

  key_t* d_keys_lhs     = thrust::raw_pointer_cast(keys_lhs.data());
  key_t* d_keys_rhs     = thrust::raw_pointer_cast(keys_rhs.data());
  key_t* d_keys_out     = thrust::raw_pointer_cast(keys_out.data());
  value_t* d_values_lhs = thrust::raw_pointer_cast(values_lhs.data());
  value_t* d_values_rhs = thrust::raw_pointer_cast(values_rhs.data());
  value_t* d_values_out = thrust::raw_pointer_cast(values_out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::dispatch(
    nullptr,
    temp_size,
    d_keys_lhs,
    d_values_lhs,
    num_items_lhs,
    d_keys_rhs,
    d_values_rhs,
    num_items_rhs,
    d_keys_out,
    d_values_out,
    compare_op_t{},
    cudaStream_t{});

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::dispatch(
      temp_storage,
      temp_size,
      d_keys_lhs,
      d_values_lhs,
      num_items_lhs,
      d_keys_rhs,
      d_values_rhs,
      num_items_rhs,
      d_keys_out,
      d_values_out,
      compare_op_t{},
      launch.get_stream());
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

// using key_types = nvbench::type_list<int32_t>;
// using value_types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(pairs, NVBENCH_TYPE_AXES(key_types, value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
