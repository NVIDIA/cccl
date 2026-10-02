// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <test_util.h>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetDifference, set_difference);

// Small key types stress the duplicate handling of the balanced merge path (many equal keys).
using key_types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

template <typename Key, typename Offset, typename CompareOp = cuda::std::less<Key>>
void test_difference(Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {})
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1, thrust::default_init);
  c2h::device_vector<Key> keys2_d(size2, thrust::default_init);
  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);

  c2h::device_vector<Key> result_d(size1 + size2, thrust::default_init);
  c2h::device_vector<Offset> num_selected_d(1, thrust::default_init);
  set_difference(
    thrust::raw_pointer_cast(keys1_d.data()),
    size1,
    thrust::raw_pointer_cast(keys2_d.data()),
    size2,
    thrust::raw_pointer_cast(result_d.data()),
    thrust::raw_pointer_cast(num_selected_d.data()),
    compare_op);

  // reference
  c2h::host_vector<Key> keys1_h = keys1_d;
  c2h::host_vector<Key> keys2_h = keys2_d;
  c2h::host_vector<Key> reference_h;
  std::set_difference(
    keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), std::back_inserter(reference_h), compare_op);

  const Offset num_selected = num_selected_d[0];
  REQUIRE(num_selected == static_cast<Offset>(reference_h.size()));
  c2h::host_vector<Key> result_h(result_d);
  result_h.resize(num_selected);
  CHECK(reference_h == result_h);
}

CUB_TEST("DeviceSetOps set difference on keys", "[set_ops][device]", CUB_SMALL, key_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;
  test_difference<key_t, offset_t>();
}

// Cover a range of input-size regimes: both empty, one side empty, single element, very asymmetric, small, medium (a
// few tiles), and large (many tiles).
CUB_TEST_CASE("DeviceSetOps set difference covers a range of input sizes", "[set_ops][device]", CUB_SMALL)
{
  using key_t               = int;
  using offset_t            = int;
  const auto [size1, size2] = GENERATE(table<int, int>({
    {0, 0}, // both empty
    {0, 137}, // first empty
    {137, 0}, // second empty
    {1, 1}, // single element each
    {1, 5000}, // very asymmetric
    {5000, 1}, // very asymmetric
    {23, 51}, // small
    {3623, 6346}, // medium
    {40000, 55000}, // large: spans many tiles
  }));
  test_difference<key_t, offset_t>(size1, size2);
}

// Regression test for a specific bug flagged in CI review: a signed 32-bit overflow in the biased binary search.
// `biased_binary_search` is called with a count up to the input size, and at shift == 9 its scale (511) times that
// count overflows a 32-bit offset once the count exceeds ~4.2M elements, corrupting the merge-path partition (which
// manifested as an out-of-bounds read). This needs more than ~4.2M keys on the first input.
CUB_TEST_CASE("DeviceSetOps set difference handles inputs above the biased-search overflow threshold",
              "[set_ops][device]",
              CUB_SMALL)
{
  test_difference<int, int>(5'000'000, 5'000'000);
}

CUB_TEST_CASE("DeviceSetOps set difference with a custom comparator", "[set_ops][device]", CUB_SMALL)
{
  using key_t    = int;
  using offset_t = int;
  test_difference<key_t, offset_t>(2000, 3000, cuda::std::greater<key_t>{});
}

CUB_TEST_CASE("DeviceSetOps set difference uses virtual shared memory for large key types",
              "[set_ops][device]",
              CUB_SMALL)
{
  // A key type large enough that the agent's per-block temporary storage exceeds the static shared-memory limit,
  // forcing the dispatch to fall back to global-memory-backed virtual shared memory.
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>;
  using offset_t = int;
  test_difference<key_t, offset_t>();
}

// Exercises both a 32-bit and a 64-bit offset type (the type deduced by the CUB API from the size arguments).
CUB_TEST("DeviceSetOps set difference supports 32-bit and 64-bit offset types",
         "[set_ops][device]",
         CUB_SMALL,
         c2h::type_list<std::int32_t, std::int64_t>)
{
  using offset_t = c2h::get<0, TestType>;
  using key_t    = int;
  test_difference<key_t, offset_t>();
}

CUB_TEST("DeviceSetOps set difference works for a large number of items",
         "[set_ops][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         CUB_SMALL,
         c2h::type_list<std::uint32_t, std::int64_t>)
try
{
  using num_keys_t = c2h::get<0, TestType>;
  using key_t      = std::int64_t;

  const num_keys_t num_keys1 = 2'400'000'000;
  // Keep the combined input size just below 2^32 for the 32-bit offset, and push it just above for the 64-bit offset.
  const num_keys_t num_keys2 = (sizeof(num_keys_t) == 4) ? num_keys_t{1'800'000'000} : num_keys_t{2'000'000'000};
  CAPTURE(c2h::type_name<num_keys_t>(), num_keys1, num_keys2);

  const auto keys1 = cuda::counting_iterator(key_t{0});
  const auto keys2 = cuda::counting_iterator(key_t{0});

  // keys1 \ keys2 == [num_keys2, num_keys1)
  const auto expected_num_selected = static_cast<std::int64_t>(num_keys1) - static_cast<std::int64_t>(num_keys2);
  const auto expected_result_it    = cuda::counting_iterator(static_cast<key_t>(num_keys2));

  auto check_result_helper = detail::large_problem_test_helper(static_cast<std::size_t>(expected_num_selected));
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  c2h::device_vector<std::int64_t> num_selected_out(1, 0);
  auto* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  set_difference(keys1, num_keys1, keys2, num_keys2, check_result_it, d_num_selected_out, cuda::std::less<key_t>{});

  REQUIRE(num_selected_out[0] == expected_num_selected);
  check_result_helper.check_all_results_correct();
}
catch (const std::bad_alloc&)
{
  SUCCEED("exceeding memory is not a failure");
}
