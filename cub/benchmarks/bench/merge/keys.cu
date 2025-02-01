// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_merge.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/sort.h>

#include <cuda/std/utility>

#include <cstdint>

#include "merge_common.cuh"
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

template <typename KeyT, typename OffsetT>
void keys(nvbench::state& state, nvbench::type_list<KeyT, OffsetT>)
{
  using key_t            = KeyT;
  using value_t          = cub::NullType;
  using key_input_it_t   = key_t*;
  using value_input_it_t = value_t*;
  using key_it_t         = key_t*;
  using value_it_t       = value_t*;
  using offset_t         = OffsetT;
  using compare_op_t     = less_t;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t>;
  using dispatch_t = cub::cub::detail::merge::
    dispatch_t<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::detail::merge::
    dispatch_t<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  // We generate data distributions in the range [0, 255], which, with lower entropy, get skewed towards 0.
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
  // point, after which the lhs is "full". We compose the rhs by selecting all items up to the pivot point that were not
  // selected for lhs and *all* items after the pivot point.
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

  // Generate increasing input range to sample from
  thrust::device_vector<key_t> increasing_input = generate(elements);
  thrust::sort(increasing_input.begin(), increasing_input.end());

  // Select lhs from input up to pivot point
  offset_t pivot_point_val = pivot_point[0];
  auto const end_lhs       = thrust::copy_if(
    increasing_input.cbegin(),
    increasing_input.cbegin() + pivot_point_val,
    rnd_selector_val.cbegin(),
    keys_lhs.begin(),
    select_lhs_op);

  // Select rhs items from input up to pivot point
  auto const end_rhs = thrust::copy_if(
    increasing_input.cbegin(),
    increasing_input.cbegin() + pivot_point_val,
    rnd_selector_val.cbegin(),
    keys_rhs.begin(),
    select_rhs_op);
  // From pivot point copy all remaining items to rhs
  thrust::copy(increasing_input.cbegin() + pivot_point_val, increasing_input.cbegin() + elements, end_rhs);

  key_t* d_keys_lhs = thrust::raw_pointer_cast(keys_lhs.data());
  key_t* d_keys_rhs = thrust::raw_pointer_cast(keys_rhs.data());
  key_t* d_keys_out = thrust::raw_pointer_cast(keys_out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::dispatch(
    nullptr,
    temp_size,
    d_keys_lhs,
    nullptr,
    num_items_lhs,
    d_keys_rhs,
    nullptr,
    num_items_rhs,
    d_keys_out,
    nullptr,
    compare_op_t{},
    cudaStream_t{});

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::dispatch(
      temp_storage,
      temp_size,
      d_keys_lhs,
      nullptr,
      num_items_lhs,
      d_keys_rhs,
      nullptr,
      num_items_rhs,
      d_keys_out,
      nullptr,
      compare_op_t{},
      launch.get_stream());
  });
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = fundamental_types;
#endif // TUNE_KeyT

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(key_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
