// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>

#include <test_util.h>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/custom_type.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetDifferencePairs, set_difference_pairs);
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetIntersectionPairs, set_intersection_pairs);
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetSymmetricDifferencePairs, set_symmetric_difference_pairs);
DECLARE_LAUNCH_WRAPPER(cub::detail::DeviceSetOps::SetUnionPairs, set_union_pairs);

// A key type paired with a value (payload) type, so key and value widths are swept together.
template <typename K, typename V>
struct type_pair
{
  using key_t   = K;
  using value_t = V;
};

template <typename>
inline constexpr bool is_custom_type_v = false;
template <template <typename> class... Ps>
inline constexpr bool is_custom_type_v<c2h::custom_type_t<Ps...>> = true;

// The output keys of a set operation on key-value pairs must equal the keys-only result, and each emitted value must be
// the value associated with its key in whichever input it came from. We tag each value with `key * 2 + source_bit`
// (0 = first input, 1 = second input): checking the tag equals `key * 2` or `key * 2 + 1` verifies the value was
// gathered for the correct element, and the source bit lets operations that must take values exclusively from the first
// input be checked precisely. Scalar values store the tag directly; a custom payload carries it in its `val` field.
template <typename ValueT, typename KeyT>
ValueT make_tagged_value(KeyT key, std::size_t source)
{
  const std::size_t tag = static_cast<std::size_t>(key) * 2 + source;
  if constexpr (is_custom_type_v<ValueT>)
  {
    ValueT value{};
    value.key = static_cast<std::size_t>(key);
    value.val = tag;
    return value;
  }
  else
  {
    return static_cast<ValueT>(tag);
  }
}

template <typename ValueT>
std::size_t value_tag(const ValueT& value)
{
  if constexpr (is_custom_type_v<ValueT>)
  {
    return value.val;
  }
  else
  {
    return static_cast<std::size_t>(value);
  }
}

template <typename KeyT, typename ValueT, typename LaunchT, typename StdOp>
void test_pairs(LaunchT launch, StdOp std_op, bool values_from_first_input_only, int size1 = 3623, int size2 = 6346)
{
  CAPTURE(c2h::type_name<KeyT>(), c2h::type_name<ValueT>(), size1, size2);

  // Keep the keys dense (plenty of duplicate runs) and small enough that the value tag key*2+1 stays representable in
  // every value type we test.
  constexpr int key_bound = 2000;

  c2h::device_vector<KeyT> keys1_d(size1, thrust::default_init);
  c2h::device_vector<KeyT> keys2_d(size2, thrust::default_init);
  c2h::gen(C2H_SEED(1), keys1_d, KeyT{0}, KeyT{key_bound});
  c2h::gen(C2H_SEED(1), keys2_d, KeyT{0}, KeyT{key_bound});
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end());
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end());

  c2h::host_vector<KeyT> keys1_h = keys1_d;
  c2h::host_vector<KeyT> keys2_h = keys2_d;
  c2h::host_vector<ValueT> values1_h(size1, thrust::default_init);
  c2h::host_vector<ValueT> values2_h(size2, thrust::default_init);
  for (int i = 0; i < size1; ++i)
  {
    values1_h[i] = make_tagged_value<ValueT>(keys1_h[i], 0); // source bit 0
  }
  for (int i = 0; i < size2; ++i)
  {
    values2_h[i] = make_tagged_value<ValueT>(keys2_h[i], 1); // source bit 1
  }
  c2h::device_vector<ValueT> values1_d = values1_h;
  c2h::device_vector<ValueT> values2_d = values2_h;

  c2h::device_vector<KeyT> keys_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<ValueT> values_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<int> num_selected_d(1, thrust::default_init);

  launch(thrust::raw_pointer_cast(keys1_d.data()),
         thrust::raw_pointer_cast(values1_d.data()),
         size1,
         thrust::raw_pointer_cast(keys2_d.data()),
         thrust::raw_pointer_cast(values2_d.data()),
         size2,
         thrust::raw_pointer_cast(keys_out_d.data()),
         thrust::raw_pointer_cast(values_out_d.data()),
         thrust::raw_pointer_cast(num_selected_d.data()),
         // pass the comparator explicitly so the stream argument injected under graph capture lines up
         cuda::std::less<>{});

  c2h::host_vector<KeyT> reference_keys;
  std_op(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), std::back_inserter(reference_keys));

  const int num_selected = num_selected_d[0];
  REQUIRE(num_selected == static_cast<int>(reference_keys.size()));

  c2h::host_vector<KeyT> keys_out_h(keys_out_d);
  keys_out_h.resize(num_selected);
  CHECK(reference_keys == keys_out_h);

  c2h::host_vector<ValueT> values_out_h(values_out_d);
  values_out_h.resize(num_selected);
  for (int i = 0; i < num_selected; ++i)
  {
    const std::size_t key_value = static_cast<std::size_t>(keys_out_h[i]);
    const std::size_t tag       = value_tag(values_out_h[i]);
    CAPTURE(i, keys_out_h[i], tag, key_value);
    // The value must belong to the emitted key: either the first-input tag (key*2) or the second-input tag (key*2+1).
    REQUIRE((tag == key_value * 2 || tag == key_value * 2 + 1));
    if (values_from_first_input_only)
    {
      REQUIRE(tag == key_value * 2);
    }
  }
}

// Runs all four key-value set operations for the given input sizes.
template <typename KeyT, typename ValueT>
void test_all_pairs(int size1 = 3623, int size2 = 6346)
{
  test_pairs<KeyT, ValueT>(
    [](auto&&... a) {
      set_difference_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_difference(a...);
    },
    /* values_from_first_input_only */ true,
    size1,
    size2);
  test_pairs<KeyT, ValueT>(
    [](auto&&... a) {
      set_intersection_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    },
    /* values_from_first_input_only */ true,
    size1,
    size2);
  test_pairs<KeyT, ValueT>(
    [](auto&&... a) {
      set_symmetric_difference_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_symmetric_difference(a...);
    },
    /* values_from_first_input_only */ false,
    size1,
    size2);
  test_pairs<KeyT, ValueT>(
    [](auto&&... a) {
      set_union_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_union(a...);
    },
    /* values_from_first_input_only */ false,
    size1,
    size2);
}

// Regression test for an out-of-bounds read of the second value range.
//
// For set difference and intersection, every emitted value comes from the first input, so the second input's values are
// never used. Thrust's `set_difference_by_key`/`set_intersection_by_key` exploit this: they do not have a second value
// range at all and pass the *first* input's values (length num_keys1) as the dummy `d_values_in2`. The agent must
// therefore never read more than num_keys1 elements from `d_values_in2` for these two operations. A previous version
// unconditionally loaded num_keys2 values from `d_values_in2`, reading past the end of the dummy buffer whenever
// num_keys2 > num_keys1. The over-read is silent in normal runs (the loaded values are discarded) but is flagged by
// compute-sanitizer memcheck, under which this test is run in CI.
template <typename LaunchT, typename StdOp>
void test_short_values2_dummy(LaunchT launch, StdOp std_op)
{
  using KeyT   = std::int32_t;
  using ValueT = std::int32_t;

  // num_keys2 deliberately much larger than num_keys1, so an unconditional load of num_keys2 values from a
  // num_keys1-length buffer reads well past its end, and across several tiles.
  const int size1 = 1000;
  const int size2 = 50000;

  c2h::device_vector<KeyT> keys1_d(size1, thrust::default_init);
  c2h::device_vector<KeyT> keys2_d(size2, thrust::default_init);
  c2h::gen(C2H_SEED(1), keys1_d, KeyT{0}, KeyT{2000});
  c2h::gen(C2H_SEED(1), keys2_d, KeyT{0}, KeyT{2000});
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end());
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end());

  // The first input's values, reused as the dummy second value range exactly as Thrust's by-key ops do.
  c2h::device_vector<ValueT> values1_d(size1, thrust::default_init);
  thrust::sequence(c2h::device_policy, values1_d.begin(), values1_d.end());

  c2h::device_vector<KeyT> keys_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<ValueT> values_out_d(size1 + size2, thrust::default_init);
  c2h::device_vector<int> num_selected_d(1, thrust::default_init);

  launch(thrust::raw_pointer_cast(keys1_d.data()),
         thrust::raw_pointer_cast(values1_d.data()),
         size1,
         thrust::raw_pointer_cast(keys2_d.data()),
         thrust::raw_pointer_cast(values1_d.data()), // dummy: only size1 long, but num_keys2 == size2 is passed below
         size2,
         thrust::raw_pointer_cast(keys_out_d.data()),
         thrust::raw_pointer_cast(values_out_d.data()),
         thrust::raw_pointer_cast(num_selected_d.data()),
         cuda::std::less<>{});

  // The keys result must still be correct; the point of the test is that producing it must not read the dummy
  // second value range out of bounds.
  c2h::host_vector<KeyT> keys1_h = keys1_d;
  c2h::host_vector<KeyT> keys2_h = keys2_d;
  c2h::host_vector<KeyT> reference_keys;
  std_op(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), std::back_inserter(reference_keys));

  const int num_selected = num_selected_d[0];
  REQUIRE(num_selected == static_cast<int>(reference_keys.size()));
  c2h::host_vector<KeyT> keys_out_h(keys_out_d);
  keys_out_h.resize(num_selected);
  CHECK(reference_keys == keys_out_h);
}

CUB_TEST_CASE("DeviceSetOps difference/intersection pairs do not read the second value range out of bounds",
              "[set_ops][device]",
              CUB_SMALL)
{
  test_short_values2_dummy(
    [](auto&&... a) {
      set_difference_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_difference(a...);
    });
  test_short_values2_dummy(
    [](auto&&... a) {
      set_intersection_pairs(static_cast<decltype(a)>(a)...);
    },
    [](auto... a) {
      std::set_intersection(a...);
    });
}

// A spread of key/value width combinations, including a non-trivial custom value type. The agent stages a whole tile of
// values in shared memory, so any value larger than ~5 bytes pushes the per-block storage past the static shared-memory
// limit and forces the dispatch onto virtual shared memory; the custom type exercises that fallback. Keys are kept to
// integer types so the value tag (key * 2 + source) stays exact.
using pair_types =
  c2h::type_list<type_pair<std::int16_t, std::int32_t>, // baseline
                 type_pair<std::int64_t, std::int32_t>, // wide key
                 type_pair<std::int16_t, std::int64_t>, // wide value
                 type_pair<std::int32_t, // custom value type -> vsmem
                           c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<16>::type>>>;

// Run every key/value combination across a range of input-size regimes: both empty, one side empty, single element,
// very asymmetric, small, medium (a few tiles), and large (many tiles). The sizes are a runtime GENERATE sweep, so they
// cost nothing extra to compile but exercise each type against every regime.
CUB_TEST("DeviceSetOps pairs across key and value types and input sizes", "[set_ops][device]", CUB_SMALL, pair_types)
{
  using key_t               = typename c2h::get<0, TestType>::key_t;
  using value_t             = typename c2h::get<0, TestType>::value_t;
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
  test_all_pairs<key_t, value_t>(size1, size2);
}
