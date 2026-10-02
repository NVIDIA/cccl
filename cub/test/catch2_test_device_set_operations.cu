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
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetIntersection, set_intersection);
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetSymmetricDifference, set_symmetric_difference);
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetUnion, set_union);

// Small key types stress the duplicate handling of the balanced merge path (many equal keys).
using key_types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

template <typename Key, typename Offset, typename LaunchT, typename StdOp, typename CompareOp = cuda::std::less<Key>>
void test_keys(LaunchT launch, StdOp std_op, Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {})
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
  launch(thrust::raw_pointer_cast(keys1_d.data()),
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
  std_op(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), std::back_inserter(reference_h), compare_op);

  const Offset num_selected = num_selected_d[0];
  REQUIRE(num_selected == static_cast<Offset>(reference_h.size()));
  c2h::host_vector<Key> result_h(result_d);
  result_h.resize(num_selected);
  CHECK(reference_h == result_h);
}

// Runs all four set operations for the given input sizes and validates each against its std reference.
template <typename Key, typename Offset>
void test_all_ops(Offset size1, Offset size2)
{
  test_keys<Key, Offset>(
    [](auto&&... a) {
      set_difference(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_difference(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    [](auto&&... a) {
      set_intersection(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    [](auto&&... a) {
      set_symmetric_difference(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_symmetric_difference(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    [](auto&&... a) {
      set_union(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_union(a...);
    },
    size1,
    size2);
}

CUB_TEST("DeviceSetOps set operations on keys", "[set_ops][device]", CUB_SMALL, key_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;

  SECTION("difference")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_difference(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_difference(a...);
      });
  }
  SECTION("intersection")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_intersection(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_intersection(a...);
      });
  }
  SECTION("symmetric_difference")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_symmetric_difference(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_symmetric_difference(a...);
      });
  }
  SECTION("union")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_union(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_union(a...);
      });
  }
}

// Cover a range of input-size regimes for every operation: both empty, one side empty, single element, very
// asymmetric, small, medium (a few tiles), and large (many tiles).
CUB_TEST_CASE("DeviceSetOps covers a range of input sizes", "[set_ops][device]", CUB_SMALL)
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
  test_all_ops<key_t, offset_t>(size1, size2);
}

CUB_TEST_CASE("DeviceSetOps set operations on keys with a custom comparator", "[set_ops][device]", CUB_SMALL)
{
  using key_t    = int;
  using offset_t = int;
  test_keys<key_t, offset_t>(
    [](auto&&... a) {
      set_intersection(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    },
    2000,
    3000,
    cuda::std::greater<key_t>{});
}

CUB_TEST_CASE("DeviceSetOps uses virtual shared memory for large key types", "[set_ops][device]", CUB_SMALL)
{
  // A key type large enough that the agent's per-block temporary storage exceeds the static shared-memory limit,
  // forcing the dispatch to fall back to global-memory-backed virtual shared memory.
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>;
  using offset_t = int;
  test_keys<key_t, offset_t>(
    [](auto&&... a) {
      set_union(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_union(a...);
    });
}

// Exercises both a 32-bit and a 64-bit offset type (the type deduced by the CUB API from the size arguments).
CUB_TEST("DeviceSetOps supports 32-bit and 64-bit offset types",
         "[set_ops][device]",
         CUB_SMALL,
         c2h::type_list<std::int32_t, std::int64_t>)
{
  using offset_t = c2h::get<0, TestType>;
  using key_t    = int;

  SECTION("intersection")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_intersection(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_intersection(a...);
      });
  }
  SECTION("union")
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_union(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_union(a...);
      });
  }
}

// Large-input helper that avoids materializing the inputs or the output. keys1 = [0, num_keys1) and
// keys2 = [0, num_keys2) (with num_keys2 <= num_keys1, so keys2 is a prefix of keys1) come from counting iterators, so
// every set operation's result is the contiguous counting range [expected_start, expected_start + expected_count). The
// output is verified by a bit-flagging tabulate iterator against that range, so only the compacted correctness flags
// (one bit per emitted key) and the dispatch's temporary storage are allocated.
template <typename Key, typename NumKeysT, typename LaunchT>
void test_op_large(
  LaunchT launch, NumKeysT num_keys1, NumKeysT num_keys2, Key expected_start, std::int64_t expected_count)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<NumKeysT>(), num_keys1, num_keys2, expected_start, expected_count);

  const auto keys1 = cuda::counting_iterator(Key{0});
  const auto keys2 = cuda::counting_iterator(Key{0});

  const auto expected_result_it = cuda::counting_iterator(expected_start);
  auto check_result_helper      = detail::large_problem_test_helper(static_cast<std::size_t>(expected_count));
  auto check_result_it          = check_result_helper.get_flagging_output_iterator(expected_result_it);

  c2h::device_vector<std::int64_t> num_selected_out(1, 0);
  auto* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  launch(keys1, num_keys1, keys2, num_keys2, check_result_it, d_num_selected_out, cuda::std::less<Key>{});

  REQUIRE(num_selected_out[0] == expected_count);
  check_result_helper.check_all_results_correct();
}

// Runs every set operation on a combined input size straddling 2^32: just below it for the 32-bit offset type the CUB
// API deduces from a 4-byte size type, and just above it for the 64-bit offset type deduced from an 8-byte size type.
CUB_TEST("DeviceSetOps works for a large number of items",
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
  const auto n1              = static_cast<std::int64_t>(num_keys1);
  const auto n2              = static_cast<std::int64_t>(num_keys2);

  // keys2 == [0, num_keys2) is a prefix of keys1 == [0, num_keys1), so each operation yields a contiguous range.
  SECTION("difference") // keys1 \ keys2 == [num_keys2, num_keys1)
  {
    test_op_large<key_t>(
      [](auto&&... a) {
        set_difference(static_cast<decltype(a)>(a)...);
      },
      num_keys1,
      num_keys2,
      static_cast<key_t>(num_keys2),
      n1 - n2);
  }
  SECTION("intersection") // keys1 ∩ keys2 == [0, num_keys2)
  {
    test_op_large<key_t>(
      [](auto&&... a) {
        set_intersection(static_cast<decltype(a)>(a)...);
      },
      num_keys1,
      num_keys2,
      key_t{0},
      n2);
  }
  SECTION("symmetric_difference") // (keys1 \ keys2) ∪ (keys2 \ keys1) == [num_keys2, num_keys1), since keys2 ⊂ keys1
  {
    test_op_large<key_t>(
      [](auto&&... a) {
        set_symmetric_difference(static_cast<decltype(a)>(a)...);
      },
      num_keys1,
      num_keys2,
      static_cast<key_t>(num_keys2),
      n1 - n2);
  }
  SECTION("union") // keys1 ∪ keys2 == [0, num_keys1)
  {
    test_op_large<key_t>(
      [](auto&&... a) {
        set_union(static_cast<decltype(a)>(a)...);
      },
      num_keys1,
      num_keys2,
      key_t{0},
      n1);
  }
}
catch (const std::bad_alloc&)
{
  SUCCEED("exceeding memory is not a failure");
}
