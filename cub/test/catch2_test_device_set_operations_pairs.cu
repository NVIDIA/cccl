// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/sort.h>

#include <algorithm>

#include <test_util.h>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetDifferencePairs, set_difference_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetIntersectionPairs, set_intersection_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetSymmetricDifferencePairs, set_symmetric_difference_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSetOps::SetUnionPairs, set_union_pairs);

// The output keys of a set operation on key-value pairs must equal the keys-only result, and each emitted value must be
// the value associated with its key in whichever input it came from. We tag values as `key * 2 + source_bit`
// (0 = first input, 1 = second input): checking each value equals `key * 2` or `key * 2 + 1` verifies it was gathered
// for the correct element, and the source bit lets operations that must take values exclusively from the first input be
// checked precisely.
template <typename LaunchT, typename StdOp>
void test_pairs(LaunchT launch, StdOp std_op, bool values_from_first_input_only)
{
  using key_t     = std::int16_t;
  using value_t   = int;
  const int size1 = 3623;
  const int size2 = 6346;

  c2h::device_vector<key_t> keys1_d(size1, thrust::default_init);
  c2h::device_vector<key_t> keys2_d(size2, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys1_d);
  c2h::gen(C2H_SEED(2), keys2_d);
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end());
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end());

  c2h::host_vector<key_t> keys1_h = keys1_d;
  c2h::host_vector<key_t> keys2_h = keys2_d;
  c2h::host_vector<value_t> values1_h(size1);
  c2h::host_vector<value_t> values2_h(size2);
  for (int i = 0; i < size1; ++i)
  {
    values1_h[i] = static_cast<value_t>(keys1_h[i]) * 2; // source bit 0
  }
  for (int i = 0; i < size2; ++i)
  {
    values2_h[i] = static_cast<value_t>(keys2_h[i]) * 2 + 1; // source bit 1
  }
  c2h::device_vector<value_t> values1_d = values1_h;
  c2h::device_vector<value_t> values2_d = values2_h;

  c2h::device_vector<key_t> keys_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<value_t> values_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<int> num_selected_d(1, thrust::default_init);

  launch(thrust::raw_pointer_cast(keys1_d.data()),
         thrust::raw_pointer_cast(values1_d.data()),
         size1,
         thrust::raw_pointer_cast(keys2_d.data()),
         thrust::raw_pointer_cast(values2_d.data()),
         size2,
         thrust::raw_pointer_cast(keys_out_d.data()),
         thrust::raw_pointer_cast(values_out_d.data()),
         thrust::raw_pointer_cast(num_selected_d.data()));

  c2h::host_vector<key_t> reference_keys;
  std_op(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), std::back_inserter(reference_keys));

  const int num_selected = num_selected_d[0];
  REQUIRE(num_selected == static_cast<int>(reference_keys.size()));

  c2h::host_vector<key_t> keys_out_h(keys_out_d);
  keys_out_h.resize(num_selected);
  CHECK(reference_keys == keys_out_h);

  c2h::host_vector<value_t> values_out_h(values_out_d);
  values_out_h.resize(num_selected);
  for (int i = 0; i < num_selected; ++i)
  {
    const value_t key_value = static_cast<value_t>(keys_out_h[i]);
    CAPTURE(i, keys_out_h[i], values_out_h[i], key_value);
    // The value must belong to the emitted key: either the first-input tag (key*2) or the second-input tag (key*2+1).
    REQUIRE((values_out_h[i] == key_value * 2 || values_out_h[i] == key_value * 2 + 1));
    if (values_from_first_input_only)
    {
      REQUIRE(values_out_h[i] == key_value * 2);
    }
  }
}

CUB_TEST_CASE("DeviceSetOps difference pairs", "[set_ops][device]", CUB_SMALL)
{
  test_pairs(
    [](auto&&... a) {
      set_difference_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_difference(a...);
    },
    /* values_from_first_input_only */ true);
}

CUB_TEST_CASE("DeviceSetOps intersection pairs", "[set_ops][device]", CUB_SMALL)
{
  test_pairs(
    [](auto&&... a) {
      set_intersection_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    },
    /* values_from_first_input_only */ true);
}

CUB_TEST_CASE("DeviceSetOps symmetric difference pairs", "[set_ops][device]", CUB_SMALL)
{
  test_pairs(
    [](auto&&... a) {
      set_symmetric_difference_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_symmetric_difference(a...);
    },
    /* values_from_first_input_only */ false);
}

CUB_TEST_CASE("DeviceSetOps union pairs", "[set_ops][device]", CUB_SMALL)
{
  test_pairs(
    [](auto&&... a) {
      set_union_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_union(a...);
    },
    /* values_from_first_input_only */ false);
}
