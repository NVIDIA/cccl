// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_adjacent_difference.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_non_trivial_runs.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/device/dispatch/tuning/tuning_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/iterator>
#include <cuda/std/functional>

#include "cub_test_macros.h"

using offset_t = int;

// Reads memory without being contiguous, so it selects the same algorithms as a synthesizing iterator except for the
// load algorithm
template <class It>
using memory_it_t = cuda::transform_iterator<cuda::std::identity, const cub::detail::it_value_t<It>*>;

using counting_it_t        = cuda::counting_iterator<int>;
using thrust_counting_it_t = thrust::counting_iterator<int>;
using transform_it_t       = cuda::transform_iterator<cuda::std::negate<>, cuda::counting_iterator<int>>;
using zip_synth_it_t       = cuda::zip_iterator<cuda::counting_iterator<int>, cuda::constant_iterator<int>>;
using zip_mixed_it_t       = cuda::zip_iterator<cuda::counting_iterator<int>, const int*>;

constexpr cuda::compute_capability ccs[] = {
  {5, 0}, {6, 0}, {7, 0}, {7, 5}, {8, 0}, {8, 6}, {8, 9}, {9, 0}, {10, 0}, {10, 3}, {10, 7}, {12, 0}};

constexpr bool is_transposing(cub::BlockLoadAlgorithm algorithm)
{
  return algorithm == cub::BLOCK_LOAD_TRANSPOSE || algorithm == cub::BLOCK_LOAD_WARP_TRANSPOSE
      || algorithm == cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
}

CUB_TEST("load_algorithm_for_input only replaces transposing loads of synthesizing inputs", "[tuning]", CUB_SMALL)
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
  STATIC_REQUIRE(load_algorithm_for_input(cub::BLOCK_LOAD_WARP_TRANSPOSE, false) == cub::BLOCK_LOAD_WARP_TRANSPOSE);
}

CUB_TEST("Transform metadata distinguishes synthesizing inputs", "[transform][tuning]", CUB_SMALL)
{
  constexpr auto counting_info = cub::detail::make_iterator_info<counting_it_t>();
  constexpr auto memory_info   = cub::detail::make_iterator_info<memory_it_t<counting_it_t>>();

  STATIC_REQUIRE(counting_info.is_synthesizing);
  STATIC_REQUIRE_FALSE(counting_info.is_contiguous);
  STATIC_REQUIRE_FALSE(memory_info.is_synthesizing);
  STATIC_REQUIRE_FALSE(memory_info.is_contiguous);
}

template <class InputIt>
cub::ScanPolicy scan_policy(cuda::compute_capability cc)
{
  return cub::detail::scan::policy_selector_from_types<InputIt, int*, int, offset_t, cuda::std::plus<>>{}(cc);
}

template <class InputIt>
void check_scan_policy(cuda::compute_capability cc, bool synthesizing)
{
  const auto policy = scan_policy<InputIt>(cc);
  const auto memory = scan_policy<memory_it_t<InputIt>>(cc);

  if (policy.algorithm == cub::ScanAlgorithm::lookahead)
  {
    // A synthesizing input may use lookahead. A non-contiguous memory iterator of the same value type may not. The
    // lookahead tuning itself matches a contiguous pointer of that value type.
    CHECK(synthesizing);
    using value_t             = cub::detail::it_value_t<InputIt>;
    const auto pointer_policy = scan_policy<const value_t*>(cc);
    CHECK(pointer_policy.algorithm == cub::ScanAlgorithm::lookahead);
    CHECK(policy.lookahead == pointer_policy.lookahead);
    CHECK(memory.algorithm == cub::ScanAlgorithm::lookback);
  }
  else
  {
    auto expected = memory;
    expected.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(expected.lookback.load_algorithm, synthesizing);
    CHECK(policy == expected);
    if (synthesizing)
    {
      CHECK_FALSE(is_transposing(policy.lookback.load_algorithm));
    }
  }
}

CUB_TEST("Scan policy selector loads synthesizing inputs directly", "[scan][tuning]", CUB_SMALL)
{
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());
    check_scan_policy<counting_it_t>(cc, true);
    check_scan_policy<thrust_counting_it_t>(cc, true);
    check_scan_policy<transform_it_t>(cc, true);
    check_scan_policy<memory_it_t<const int*>>(cc, false);
  }
}

template <class InputIt, class FlagsIt>
cub::SelectPolicy select_policy(cuda::compute_capability cc)
{
  return cub::detail::select::policy_selector_from_types<InputIt, FlagsIt, int*, offset_t, cub::SelectImpl::Select>{}(
    cc);
}

template <class InputIt, class FlagsIt, class MemoryFlagsIt>
void check_select_policy(cuda::compute_capability cc, bool synthesizing)
{
  auto expected = select_policy<memory_it_t<InputIt>, MemoryFlagsIt>(cc);
  expected.lookback.load_algorithm =
    cub::detail::load_algorithm_for_input(expected.lookback.load_algorithm, synthesizing);

  const auto policy = select_policy<InputIt, FlagsIt>(cc);
  CHECK(policy == expected);
  if (synthesizing)
  {
    CHECK_FALSE(is_transposing(policy.lookback.load_algorithm));
  }
}

CUB_TEST("Select policy selector loads synthesizing inputs directly", "[select][tuning]", CUB_SMALL)
{
  using no_flags_t     = cub::NullType*;
  using memory_flags_t = const char*;
  using synth_flags_t  = cuda::constant_iterator<char>;
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());
    check_select_policy<counting_it_t, no_flags_t, no_flags_t>(cc, true);
    check_select_policy<thrust_counting_it_t, no_flags_t, no_flags_t>(cc, true);
    check_select_policy<zip_synth_it_t, no_flags_t, no_flags_t>(cc, true);
    check_select_policy<zip_mixed_it_t, no_flags_t, no_flags_t>(cc, false);
    check_select_policy<counting_it_t, synth_flags_t, memory_flags_t>(cc, true);

    // Items and flags share the load algorithm, so flags loaded from memory keep the tuned algorithm
    check_select_policy<counting_it_t, memory_flags_t, memory_flags_t>(cc, false);
  }
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

CUB_TEST("RLE lookback policies load synthesizing inputs directly", "[rle][tuning]", CUB_SMALL)
{
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());

    auto expected_encode = rle_encode_policy<memory_it_t<counting_it_t>>(cc);
    expected_encode.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(expected_encode.lookback.load_algorithm, true);
    const auto encode = rle_encode_policy<counting_it_t>(cc);
    CHECK(encode == expected_encode);
    CHECK(rle_encode_policy<thrust_counting_it_t>(cc) == encode);
    CHECK(encode.algorithm == cub::RleAlgorithm::lookback);
    CHECK_FALSE(is_transposing(encode.lookback.load_algorithm));

    auto expected_non_trivial = rle_non_trivial_runs_policy<memory_it_t<counting_it_t>>(cc);
    expected_non_trivial.lookback.load_algorithm =
      cub::detail::load_algorithm_for_input(expected_non_trivial.lookback.load_algorithm, true);
    const auto non_trivial = rle_non_trivial_runs_policy<counting_it_t>(cc);
    CHECK(non_trivial == expected_non_trivial);
    CHECK(rle_non_trivial_runs_policy<thrust_counting_it_t>(cc) == non_trivial);
    CHECK_FALSE(is_transposing(non_trivial.lookback.load_algorithm));
  }
}

template <class InputIt>
cub::AdjacentDifferencePolicy adjacent_difference_policy(cuda::compute_capability cc)
{
  return cub::detail::adjacent_difference::policy_selector_from_types<InputIt, false>{}(cc);
}

template <class InputIt>
void check_adjacent_difference_policy(cuda::compute_capability cc, bool synthesizing)
{
  auto expected           = adjacent_difference_policy<memory_it_t<InputIt>>(cc);
  expected.load_algorithm = cub::detail::load_algorithm_for_input(expected.load_algorithm, synthesizing);

  const auto policy = adjacent_difference_policy<InputIt>(cc);
  CHECK(policy == expected);
  if (synthesizing)
  {
    CHECK_FALSE(is_transposing(policy.load_algorithm));
  }
}

// Inclusive sum of 0, 1, 2, ... is the triangular numbers. num_items stays inside int32.
template <class InputIt>
void check_inclusive_sum(InputIt input, int num_items, int (*expected_at)(int))
{
  thrust::device_vector<int> out(static_cast<size_t>(num_items));
  REQUIRE(cub::DeviceScan::InclusiveSum(input, thrust::raw_pointer_cast(out.data()), num_items) == cudaSuccess);
  const thrust::host_vector<int> host_out = out;
  for (int i = 0; i < num_items; ++i)
  {
    if (host_out[i] != expected_at(i))
    {
      CAPTURE(i, host_out[i], expected_at(i));
      FAIL("inclusive sum mismatch");
    }
  }
}

CUB_TEST("InclusiveSum of synthesizing iterators", "[scan][device]", CUB_SMALL)
{
  // Larger than one lookahead tile on every tuned architecture, and small enough that i * (i + 1) / 2 fits in int.
  constexpr int num_items = 20000;
  check_inclusive_sum(cuda::counting_iterator<int>{0}, num_items, [](int i) {
    return i * (i + 1) / 2;
  });
  check_inclusive_sum(thrust::make_counting_iterator(0), num_items, [](int i) {
    return i * (i + 1) / 2;
  });
  check_inclusive_sum(
    cuda::transform_iterator{cuda::counting_iterator<int>{0}, cuda::std::negate<>{}}, num_items, [](int i) {
      return -(i * (i + 1) / 2);
    });
}

CUB_TEST("Adjacent difference policy selector loads synthesizing inputs directly",
         "[adjacent_difference][tuning]",
         CUB_SMALL)
{
  for (const auto cc : ccs)
  {
    CAPTURE(cc.get());
    check_adjacent_difference_policy<counting_it_t>(cc, true);
    check_adjacent_difference_policy<thrust_counting_it_t>(cc, true);
    check_adjacent_difference_policy<zip_synth_it_t>(cc, true);
    check_adjacent_difference_policy<zip_mixed_it_t>(cc, false);
  }
}
