// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>

#include <test_util.h>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetDifference, set_difference);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetIntersection, set_intersection);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetSymmetricDifference, set_symmetric_difference);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetUnion, set_union);

// Small key types stress the duplicate handling of the balanced merge path (many equal keys).
using key_types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

template <typename Key, typename Offset, typename LaunchT, typename StdOp, typename CompareOp = cuda::std::less<Key>>
void test_keys(
  c2h::seed_t seed, LaunchT launch, StdOp std_op, Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {})
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2, seed.get());

  c2h::device_vector<Key> keys1_d(size1, thrust::default_init);
  c2h::device_vector<Key> keys2_d(size2, thrust::default_init);
  // The two inputs use independent seeds so they are not correlated.
  c2h::gen(seed, keys1_d);
  c2h::gen(c2h::seed_t{seed.get() + 1}, keys2_d);
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
void test_all_ops(c2h::seed_t seed, Offset size1, Offset size2)
{
  test_keys<Key, Offset>(
    seed,
    [](auto&&... a) {
      set_difference(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_difference(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    seed,
    [](auto&&... a) {
      set_intersection(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    seed,
    [](auto&&... a) {
      set_symmetric_difference(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_symmetric_difference(a...);
    },
    size1,
    size2);
  test_keys<Key, Offset>(
    seed,
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
      C2H_SEED(2),
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
      C2H_SEED(2),
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
      C2H_SEED(2),
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
      C2H_SEED(2),
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
  using key_t    = int;
  using offset_t = int;
  // A single seed at the test-case level: threading it through avoids multiplying Catch2 generators across the
  // size/operation loop below (each C2H_SEED introduces a generator dimension).
  const c2h::seed_t seed = C2H_SEED(1);
  for (auto [s1, s2] : {
         std::pair{0, 0}, // both empty
         std::pair{0, 137}, // first empty
         std::pair{137, 0}, // second empty
         std::pair{1, 1}, // single element each
         std::pair{1, 5000}, // very asymmetric
         std::pair{5000, 1}, // very asymmetric
         std::pair{23, 51}, // small
         std::pair{3623, 6346}, // medium
         std::pair{40000, 55000}, // large: spans many tiles
       })
  {
    test_all_ops<key_t, offset_t>(seed, s1, s2);
  }
}

CUB_TEST_CASE("DeviceSetOps set operations on keys with a custom comparator", "[set_ops][device]", CUB_SMALL)
{
  using key_t    = int;
  using offset_t = int;
  test_keys<key_t, offset_t>(
    C2H_SEED(2),
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
    C2H_SEED(2),
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
      C2H_SEED(2),
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
      C2H_SEED(2),
      [](auto&&... a) {
        set_union(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_union(a...);
      });
  }
}
