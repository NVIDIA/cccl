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
void test_keys(LaunchT launch, StdOp std_op, Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {})
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1, thrust::default_init);
  c2h::device_vector<Key> keys2_d(size2, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys1_d);
  c2h::gen(C2H_SEED(2), keys2_d);
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

CUB_TEST_CASE("DeviceSetOps handles empty and single-tile inputs", "[set_ops][device]", CUB_SMALL)
{
  using key_t    = int;
  using offset_t = int;
  for (auto [s1, s2] : {std::pair{0, 0}, std::pair{0, 100}, std::pair{100, 0}, std::pair{1, 1}, std::pair{50, 4000}})
  {
    test_keys<key_t, offset_t>(
      [](auto&&... a) {
        set_union(static_cast<decltype(a)>(a)...);
      },
      [](auto... a) {
        std::set_union(a...);
      },
      s1,
      s2);
  }
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

CUB_TEST_CASE("DeviceSetOps spans multiple tiles", "[set_ops][device]", CUB_SMALL)
{
  using key_t    = int;
  using offset_t = int;
  test_keys<key_t, offset_t>(
    [](auto&&... a) {
      set_symmetric_difference(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_symmetric_difference(a...);
    },
    100000,
    130000);
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
