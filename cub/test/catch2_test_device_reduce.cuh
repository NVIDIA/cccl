// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/thread/thread_operators.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/iterator>

#include <nv/target>

#include <iostream>
#include <numeric>
#include <type_traits>

#include "catch2_test_numeric_helpers.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>
#include <c2h/test_util_vec.h>

template <typename InputItT,
          typename OffsetItT,
          typename SizeItT,
          typename ReductionOpT,
          typename InitValueT,
          typename ResultOutItT>
inline void compute_host_reference(
  InputItT h_in,
  OffsetItT h_offsets,
  SizeItT h_sizes_begin,
  std::size_t num_segments,
  ReductionOpT reduction_op,
  InitValueT init,
  ResultOutItT h_data_out)
{
  for (std::size_t segment = 0; segment < num_segments; segment++)
  {
    auto seg_begin = h_in + h_offsets[segment];
    auto seg_end   = seg_begin + h_sizes_begin[segment];
    // TODO Should this be using cub accumulator t?
    h_data_out[segment] =
      static_cast<cub::detail::it_value_t<ResultOutItT>>(std::accumulate(seg_begin, seg_end, init, reduction_op));
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification taking an
 * arbitrary host-accessible input iterator.
 */
template <typename InputItT, typename ReductionOpT, typename AccumulatorT>
inline AccumulatorT
compute_single_problem_reference(InputItT h_in_begin, InputItT h_in_end, ReductionOpT reduction_op, AccumulatorT init)
{
  constexpr std::size_t num_segments = 1;
  c2h::host_vector<AccumulatorT> h_results(num_segments);

  compute_host_reference(
    h_in_begin,
    cuda::constant_iterator(0),
    cuda::constant_iterator(cuda::std::distance(h_in_begin, h_in_end)),
    num_segments,
    reduction_op,
    init,
    h_results.begin());

  return *h_results.begin();
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector.
 */
template <typename ItemT, typename ReductionOpT, typename AccumulatorT>
inline AccumulatorT
compute_single_problem_reference(const c2h::device_vector<ItemT>& d_in, ReductionOpT reduction_op, AccumulatorT init)
{
  constexpr std::size_t num_segments = 1;
  const c2h::host_vector<ItemT> h_items(d_in);
  const c2h::host_vector<AccumulatorT> h_results(num_segments);

  return compute_single_problem_reference(h_items.cbegin(), h_items.cend(), reduction_op, init);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ReductionOpT, typename AccumulatorT, typename ResultItT>
void compute_segmented_problem_reference(
  const c2h::device_vector<ItemT>& d_in,
  const c2h::device_vector<OffsetT>& d_offsets,
  ReductionOpT reduction_op,
  AccumulatorT init,
  ResultItT h_results)
{
  const c2h::host_vector<ItemT> h_items(d_in);
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  auto offsets_it   = h_offsets.cbegin();
  auto seg_sizes_it = cuda::transform_iterator(cuda::counting_iterator(std::size_t{0}), [offsets_it](std::size_t i) {
    return offsets_it[i + 1] - offsets_it[i];
  });
  const std::size_t num_segments = h_offsets.size() - 1;

  compute_host_reference(
    h_items.cbegin(), h_offsets.cbegin(), seg_sizes_it, num_segments, reduction_op, init, h_results);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * host-accessible input iterator and a c2h::device_vector of offsets into the segments.
 */
template <typename InputItT, typename OffsetT, typename ReductionOpT, typename AccumulatorT, typename ResultItT>
void compute_segmented_problem_reference(
  InputItT in_it,
  const c2h::device_vector<OffsetT>& d_offsets,
  ReductionOpT reduction_op,
  AccumulatorT init,
  ResultItT h_results)
{
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  auto offsets_it   = h_offsets.cbegin();
  auto seg_sizes_it = cuda::transform_iterator(cuda::counting_iterator(std::size_t{0}), [offsets_it](std::size_t i) {
    return offsets_it[i + 1] - offsets_it[i];
  });
  const std::size_t num_segments = h_offsets.size() - 1;

  compute_host_reference(in_it, h_offsets.cbegin(), seg_sizes_it, num_segments, reduction_op, init, h_results);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ResultItT>
void compute_segmented_argmin_reference(
  const c2h::device_vector<ItemT>& d_in, const c2h::device_vector<OffsetT>& d_offsets, ResultItT h_results)
{
  const c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const auto num_segments = h_offsets.size() - 1;
  for (std::size_t seg = 0; seg < num_segments; seg++)
  {
    if (h_offsets[seg] >= h_offsets[seg + 1])
    {
      h_results[seg] = {1, ::cuda::std::numeric_limits<ItemT>::max()};
    }
    else
    {
      auto expected_result_it =
        std::min_element(h_items.cbegin() + h_offsets[seg], h_items.cbegin() + h_offsets[seg + 1]);
      const int result_offset =
        static_cast<int>(cuda::std::distance((h_items.cbegin() + h_offsets[seg]), expected_result_it));
      h_results[seg] = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ResultItT>
void compute_segmented_argmax_reference(
  const c2h::device_vector<ItemT>& d_in, const c2h::device_vector<OffsetT>& d_offsets, ResultItT h_results)
{
  const c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const auto num_segments = h_offsets.size() - 1;
  for (std::size_t seg = 0; seg < num_segments; seg++)
  {
    if (h_offsets[seg] >= h_offsets[seg + 1])
    {
      h_results[seg] = {1, ::cuda::std::numeric_limits<ItemT>::lowest()};
    }
    else
    {
      auto expected_result_it =
        std::max_element(h_items.cbegin() + h_offsets[seg], h_items.cbegin() + h_offsets[seg + 1]);
      const int result_offset =
        static_cast<int>(cuda::std::distance((h_items.cbegin() + h_offsets[seg]), expected_result_it));
      h_results[seg] = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items, num_segments and segment_size.
 */
template <typename ItemT, typename ReductionOpT, typename AccumulatorT, typename ResultItT>
void compute_fixed_size_segmented_problem_reference(
  const c2h::device_vector<ItemT>& d_in,
  const int num_segments,
  const int segment_size,
  ReductionOpT reduction_op,
  AccumulatorT init,
  ResultItT h_results)
{
  const c2h::host_vector<ItemT> h_items(d_in);
  auto h_begin = h_items.cbegin();

  for (int segment = 0; segment < num_segments; segment++)
  {
    auto seg_begin = h_begin + static_cast<long>(segment) * segment_size;
    auto seg_end   = seg_begin + segment_size;
    h_results[segment] =
      static_cast<cub::detail::it_value_t<ResultItT>>(std::accumulate(seg_begin, seg_end, init, reduction_op));
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items, num_segments and segment_size.
 */
template <typename ItemT, typename ResultItT>
void compute_fixed_size_segmented_argmax_reference(
  const c2h::device_vector<ItemT>& d_in, const int num_segments, const int segment_size, ResultItT h_results)
{
  c2h::host_vector<ItemT> h_items(d_in);
  auto h_begin = h_items.begin();

  for (int seg = 0; seg < num_segments; seg++)
  {
    if (segment_size == 0)
    {
      h_results[seg] = {1, ::cuda::std::numeric_limits<ItemT>::lowest()};
    }
    else
    {
      auto seg_begin          = h_begin + static_cast<long>(seg) * segment_size;
      auto seg_end            = seg_begin + segment_size;
      auto expected_result_it = std::max_element(seg_begin, seg_end);
      const int result_offset = static_cast<int>(::cuda::std::distance((seg_begin), expected_result_it));
      h_results[seg]          = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items, num_segments and segment_size.
 */
template <typename ItemT, typename ResultItT>
void compute_fixed_size_segmented_argmin_reference(
  const c2h::device_vector<ItemT>& d_in, const int num_segments, const int segment_size, ResultItT h_results)
{
  c2h::host_vector<ItemT> h_items(d_in);
  auto h_begin = h_items.begin();

  for (int seg = 0; seg < num_segments; seg++)
  {
    if (segment_size == 0)
    {
      h_results[seg] = {1, ::cuda::std::numeric_limits<ItemT>::lowest()};
    }
    else
    {
      auto seg_begin          = h_begin + static_cast<long>(seg) * segment_size;
      auto seg_end            = seg_begin + segment_size;
      auto expected_result_it = std::min_element(seg_begin, seg_end);
      const int result_offset = static_cast<int>(::cuda::std::distance((seg_begin), expected_result_it));
      h_results[seg]          = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for unique keys (i.e., collapsing each
 * run of equal keys into a single key).
 */
template <typename InputItT, typename OutputItT>
inline OutputItT compute_unique_keys_reference(InputItT h_in_begin, std::size_t num_keys, OutputItT h_out_it)
{
  if (num_keys == 0)
  {
    return h_out_it;
  }
  *h_out_it++ = h_in_begin[0];
  for (std::size_t i = 1; i < num_keys; i++)
  {
    if (!(h_in_begin[i - 1] == h_in_begin[i]))
    {
      *h_out_it = h_in_begin[i];
      h_out_it++;
    }
  }
  return h_out_it;
}

/**
 * @brief Helper function to compute the reference solution for unique keys (i.e., collapsing each
 * run of equal keys into a single key).
 */
template <typename ItemT>
inline c2h::host_vector<ItemT> compute_unique_keys_reference(const c2h::device_vector<ItemT>& d_keys)
{
  const c2h::host_vector<ItemT> h_keys(d_keys);
  c2h::host_vector<ItemT> h_unique_keys_out(d_keys.size());

  auto end_it = compute_unique_keys_reference(h_keys.cbegin(), h_keys.size(), h_unique_keys_out.begin());
  h_unique_keys_out.resize(cuda::std::distance(h_unique_keys_out.begin(), end_it));
  return h_unique_keys_out;
}

/**
 * @brief Helper class template to facilitate specifying input/output type pairs along with the key
 * type for reduce-by-key algorithms.
 */
template <typename InputT, typename OutputT = InputT, typename KeyT = std::int32_t>
struct type_triple
{
  using input_t  = InputT;
  using output_t = OutputT;
  using key_t    = KeyT;
};

/**
 * @brief Helper class template to facilitate specifying input/output type pairs.
 */
template <typename InputT, typename OutputT = InputT>
struct type_pair
{
  using input_t  = InputT;
  using output_t = OutputT;
};

/**
 * @brief Helper class template to facilitate accessing types specified by type-parameterized tests.
 */
template <typename TestType>
struct params_t
{
  using type_pair_t = typename c2h::get<0, TestType>;
  using item_t      = typename type_pair_t::input_t;
  using output_t    = typename type_pair_t::output_t;
};
