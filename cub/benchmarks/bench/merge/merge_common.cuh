// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/sort.h>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
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
  struct policy_t : cub::ChainedPolicy<500, policy_t, policy_t>
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

struct select_if_less_than_t
{
  bool negate;
  uint8_t threshold;

  __device__ __forceinline__ bool operator()(uint8_t val) const
  {
    return negate ? !(val < threshold) : val < threshold;
  }
};

template <typename OffsetT>
struct write_pivot_point_t
{
  OffsetT threshold;
  OffsetT* pivot_point;

  __device__ void operator()(OffsetT output_index, OffsetT input_index) const
  {
    if (output_index == threshold)
    {
      *pivot_point = input_index;
    }
  }
};

template <typename KeyT>
std::pair<thrust::device_vector<KeyT>, thrust::device_vector<KeyT>>
generate_lhs_rhs(std::size_t num_items_lhs, std::size_t num_items_rhs, bit_entropy entropy)
{
  using offset_t = std::size_t;

  const auto elements = num_items_lhs + num_items_rhs;

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
  if (num_items_selected_into_lhs < num_items_lhs)
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
  auto counting_it = thrust::make_counting_iterator(offset_t{0});
  thrust::copy_if(
    counting_it,
    counting_it + elements,
    rnd_selector_val.begin(),
    thrust::make_tabulate_output_iterator(write_pivot_point_t<offset_t>{
      static_cast<offset_t>(num_items_lhs), thrust::raw_pointer_cast(pivot_point.data())}),
    select_lhs_op);

  thrust::device_vector<KeyT> keys_lhs(num_items_lhs);
  thrust::device_vector<KeyT> keys_rhs(num_items_rhs);

  thrust::device_vector<KeyT> increasing_input = generate(elements);
  thrust::sort(increasing_input.begin(), increasing_input.end());

  offset_t pivot_point_val = pivot_point[0];
  auto const end_lhs       = thrust::copy_if(
    increasing_input.cbegin(),
    increasing_input.cbegin() + pivot_point_val,
    rnd_selector_val.cbegin(),
    keys_lhs.begin(),
    select_lhs_op);

  auto const end_rhs = thrust::copy_if(
    increasing_input.cbegin(),
    increasing_input.cbegin() + pivot_point_val,
    rnd_selector_val.cbegin(),
    keys_rhs.begin(),
    select_rhs_op);
  thrust::copy(increasing_input.cbegin() + pivot_point_val, increasing_input.cbegin() + elements, end_rhs);

  return {keys_lhs, keys_rhs};
}
