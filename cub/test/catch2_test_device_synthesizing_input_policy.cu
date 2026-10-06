// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/dispatch/tuning/tuning_adjacent_difference.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_non_trivial_runs.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_unique_by_key.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/iterator>
#include <cuda/std/functional>

#include "cub_test_macros.h"

using offset_t = int;

template <class It>
using memory_it_t = cuda::transform_iterator<cuda::std::identity, const cub::detail::it_value_t<It>*>;

using counting_it_t        = cuda::counting_iterator<int>;
using thrust_counting_it_t = thrust::counting_iterator<int>;
using synth_flags_it_t     = cuda::constant_iterator<char>;
using memory_flags_it_t    = const char*;

constexpr cuda::compute_capability ccs[] = {
  {5, 0}, {6, 0}, {7, 0}, {7, 5}, {8, 0}, {8, 6}, {8, 9}, {9, 0}, {10, 0}, {10, 3}, {10, 7}, {12, 0}};

constexpr bool is_transposing(cub::BlockLoadAlgorithm algorithm)
{
  return algorithm == cub::BLOCK_LOAD_TRANSPOSE || algorithm == cub::BLOCK_LOAD_WARP_TRANSPOSE
      || algorithm == cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
}

CUB_TEST("load_algorithm_for_input only replaces transposing loads", "[tuning]", CUB_SMALL)
{
  using cub::detail::load_algorithm_for_input;
  constexpr cub::BlockLoadAlgorithm algorithms[] = {
    cub::BLOCK_LOAD_DIRECT,
    cub::BLOCK_LOAD_STRIPED,
    cub::BLOCK_LOAD_VECTORIZE,
    cub::BLOCK_LOAD_TRANSPOSE,
    cub::BLOCK_LOAD_WARP_TRANSPOSE,
    cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED};

  for (const auto algorithm : algorithms)
  {
    CAPTURE(algorithm);
    CHECK(load_algorithm_for_input(algorithm, false) == algorithm);
    CHECK(load_algorithm_for_input(algorithm, true)
          == (is_transposing(algorithm) ? cub::BLOCK_LOAD_DIRECT : algorithm));
  }

  STATIC_REQUIRE(load_algorithm_for_input(cub::BLOCK_LOAD_WARP_TRANSPOSE, true) == cub::BLOCK_LOAD_DIRECT);
}

template <class InputIt>
cub::AdjacentDifferencePolicy adjacent_difference_policy(cuda::compute_capability cc)
{
  return cub::detail::adjacent_difference::policy_selector_from_types<InputIt, false>{}(cc);
}

template <class InputIt>
cub::ScanPolicy scan_policy(cuda::compute_capability cc)
{
  return cub::detail::scan::policy_selector_from_types<InputIt, int*, int, offset_t, cuda::std::plus<>>{}(cc);
}

template <class InputIt>
cub::SegmentedScanPolicy segmented_scan_policy(cuda::compute_capability cc)
{
  return cub::detail::segmented_scan::policy_selector_from_types<int, InputIt>{}(cc);
}

template <class InputIt>
cub::ThreeWayPartitionPolicy three_way_partition_policy(cuda::compute_capability cc)
{
  return cub::detail::three_way_partition::policy_selector_from_types<int, offset_t, InputIt>{}(cc);
}

template <class InputIt>
cub::RleEncodePolicy rle_encode_policy(cuda::compute_capability cc)
{
  return cub::detail::rle::encode::policy_selector_from_types<int, int, InputIt, int*, int*, int*, offset_t>{}(cc);
}

template <class InputIt>
cub::RleNonTrivialRunsPolicy rle_non_trivial_runs_policy(cuda::compute_capability cc)
{
  return cub::detail::rle::non_trivial_runs::policy_selector_from_types<int, int, InputIt>{}(cc);
}

CUB_TEST("Single-input policies load synthesizing inputs directly", "[tuning]", CUB_SMALL)
{
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());

    auto adjacent_expected           = adjacent_difference_policy<memory_it_t<counting_it_t>>(cc);
    adjacent_expected.load_algorithm = cub::detail::load_algorithm_for_input(adjacent_expected.load_algorithm, true);
    CHECK(adjacent_difference_policy<counting_it_t>(cc) == adjacent_expected);
    CHECK(adjacent_difference_policy<thrust_counting_it_t>(cc) == adjacent_expected);
    CHECK_FALSE(is_transposing(adjacent_expected.load_algorithm));

    auto scan_expected = scan_policy<memory_it_t<counting_it_t>>(cc);
    scan_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(scan_expected.lookback.load_algorithm, true);
    CHECK(scan_policy<counting_it_t>(cc) == scan_expected);
    CHECK(scan_policy<thrust_counting_it_t>(cc) == scan_expected);
    CHECK_FALSE(is_transposing(scan_expected.lookback.load_algorithm));

    auto segmented_expected = segmented_scan_policy<memory_it_t<counting_it_t>>(cc);
    segmented_expected.block.load_algorithm =
      cub::detail::load_algorithm_for_input(segmented_expected.block.load_algorithm, true);
    CHECK(segmented_scan_policy<counting_it_t>(cc) == segmented_expected);
    CHECK_FALSE(is_transposing(segmented_expected.block.load_algorithm));

    auto partition_expected = three_way_partition_policy<memory_it_t<counting_it_t>>(cc);
    partition_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(partition_expected.lookback.load_algorithm, true);
    CHECK(three_way_partition_policy<counting_it_t>(cc) == partition_expected);
    CHECK_FALSE(is_transposing(partition_expected.lookback.load_algorithm));

    auto encode_expected = rle_encode_policy<memory_it_t<counting_it_t>>(cc);
    encode_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(encode_expected.lookback.load_algorithm, true);
    CHECK(rle_encode_policy<counting_it_t>(cc) == encode_expected);
    CHECK_FALSE(is_transposing(encode_expected.lookback.load_algorithm));

    auto non_trivial_expected = rle_non_trivial_runs_policy<memory_it_t<counting_it_t>>(cc);
    non_trivial_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(non_trivial_expected.lookback.load_algorithm, true);
    CHECK(rle_non_trivial_runs_policy<counting_it_t>(cc) == non_trivial_expected);
    CHECK_FALSE(is_transposing(non_trivial_expected.lookback.load_algorithm));
  }
}

template <class InputIt, class FlagsIt>
cub::SelectPolicy select_policy(cuda::compute_capability cc)
{
  return cub::detail::select::policy_selector_from_types<InputIt, FlagsIt, int*, offset_t, cub::SelectImpl::Select>{}(
    cc);
}

template <class KeyIt, class ValueIt>
cub::ScanByKeyPolicy scan_by_key_policy(cuda::compute_capability cc)
{
  return cub::detail::scan_by_key::policy_selector_from_types<int, int, int, cuda::std::plus<>, KeyIt, ValueIt>{}(cc);
}

template <class KeyIt, class ValueIt>
cub::ReduceByKeyPolicy reduce_by_key_policy(cuda::compute_capability cc)
{
  return cub::detail::reduce_by_key::policy_selector_from_types<cuda::std::plus<>, int, int, KeyIt, ValueIt>{}(cc);
}

template <class KeyIt, class ValueIt>
cub::UniqueByKeyPolicy unique_by_key_policy(cuda::compute_capability cc)
{
  return cub::detail::unique_by_key::policy_selector_from_types<int, int, KeyIt, ValueIt>{}(cc);
}

CUB_TEST("Shared-load policies require every input to synthesize", "[tuning]", CUB_SMALL)
{
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());

    using no_flags_it_t           = cub::NullType*;
    auto select_no_flags_expected = select_policy<memory_it_t<counting_it_t>, no_flags_it_t>(cc);
    select_no_flags_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(select_no_flags_expected.lookback.load_algorithm, true);
    CHECK(select_policy<counting_it_t, no_flags_it_t>(cc) == select_no_flags_expected);
    CHECK_FALSE(is_transposing(select_no_flags_expected.lookback.load_algorithm));

    auto select_expected = select_policy<memory_it_t<counting_it_t>, memory_flags_it_t>(cc);
    select_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(select_expected.lookback.load_algorithm, true);
    CHECK(select_policy<counting_it_t, synth_flags_it_t>(cc) == select_expected);
    CHECK_FALSE(is_transposing(select_expected.lookback.load_algorithm));
    CHECK(select_policy<counting_it_t, memory_flags_it_t>(cc)
          == select_policy<memory_it_t<counting_it_t>, memory_flags_it_t>(cc));

    auto scan_by_key_expected = scan_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc);
    scan_by_key_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(scan_by_key_expected.lookback.load_algorithm, true);
    CHECK(scan_by_key_policy<counting_it_t, counting_it_t>(cc) == scan_by_key_expected);
    CHECK_FALSE(is_transposing(scan_by_key_expected.lookback.load_algorithm));
    CHECK(scan_by_key_policy<counting_it_t, memory_it_t<counting_it_t>>(cc)
          == scan_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc));

    auto reduce_by_key_expected = reduce_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc);
    reduce_by_key_expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(reduce_by_key_expected.lookback.load_algorithm, true);
    CHECK(reduce_by_key_policy<counting_it_t, counting_it_t>(cc) == reduce_by_key_expected);
    CHECK_FALSE(is_transposing(reduce_by_key_expected.lookback.load_algorithm));
    CHECK(reduce_by_key_policy<counting_it_t, memory_it_t<counting_it_t>>(cc)
          == reduce_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc));

    auto unique_by_key_expected = unique_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc);
    unique_by_key_expected.load_algorithm =
      cub::detail::load_algorithm_for_input(unique_by_key_expected.load_algorithm, true);
    CHECK(unique_by_key_policy<counting_it_t, counting_it_t>(cc) == unique_by_key_expected);
    CHECK_FALSE(is_transposing(unique_by_key_expected.load_algorithm));
    CHECK(unique_by_key_policy<counting_it_t, memory_it_t<counting_it_t>>(cc)
          == unique_by_key_policy<memory_it_t<counting_it_t>, memory_it_t<counting_it_t>>(cc));
  }
}
