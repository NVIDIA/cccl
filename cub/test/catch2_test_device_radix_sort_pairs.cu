// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/memory.h>

#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/catch2_allocation_failure.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairsDescending, sort_pairs_descending);

using custom_value_t = c2h::custom_type_t<c2h::equal_comparable_t>;
using value_types    = c2h::type_list<cuda::std::uint8_t, cuda::std::uint64_t, custom_value_t>;

// cub::detail::ChooseOffsetsT only selected 32/64 bit unsigned types:
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

CUB_TEST(
  "DeviceRadixSort::SortPairs: Basic testing", "[pairs][radix][sort][device]", CUB_SMALL, value_types, num_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using value_t     = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  constexpr num_items_t min_num_items = 1 << 5;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(5, random(min_num_items, max_num_items)));

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<value_t> out_values(num_items);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  const bool is_descending = GENERATE(false, true);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(thrust::raw_pointer_cast(in_keys.data()),
               thrust::raw_pointer_cast(out_keys.data()),
               thrust::raw_pointer_cast(in_values.data()),
               thrust::raw_pointer_cast(out_values.data()),
               num_items,
               begin_bit<key_t>(),
               end_bit<key_t>());
  }

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}

CUB_TEST("DeviceRadixSort::SortPairs: DoubleBuffer API", "[pairs][radix][sort][device]", CUB_SMALL, value_types)
{
  using key_t   = cuda::std::uint32_t;
  using value_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<value_t> out_values(num_items);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  const bool is_descending = GENERATE(false, true);

  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  cub::DoubleBuffer<value_t> value_buffer(
    thrust::raw_pointer_cast(in_values.data()), thrust::raw_pointer_cast(out_values.data()));

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, key_buffer, value_buffer, num_items, begin_bit<key_t>(), end_bit<key_t>());

  key_buffer.selector   = action.selector();
  value_buffer.selector = action.selector();
  action.finalize();

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  auto& keys   = key_buffer.selector == 0 ? in_keys : out_keys;
  auto& values = value_buffer.selector == 0 ? in_values : out_values;

  REQUIRE(ref_keys == keys);
  REQUIRE(ref_values == values);
}

template <typename KeyT, typename ValueT, typename NumItemsT>
void do_large_offset_test(std::size_t num_items)
{
  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, is_descending);

  try
  {
    large_array_sort_helper<KeyT, ValueT> arrays;
    arrays.initialize_for_stable_pair_sort(C2H_SEED(1), num_items, is_descending);

    TIME(c2h::cpu_timer timer);

    double_buffer_sort_t action(is_descending);
    action.initialize();
    const NumItemsT typed_num_items = static_cast<NumItemsT>(num_items);
    launch(action, arrays.keys_buffer, arrays.values_buffer, typed_num_items, begin_bit<KeyT>(), end_bit<KeyT>());

    TIME(timer.print_elapsed_seconds_and_reset("Device sort"));

    arrays.keys_buffer.selector   = action.selector();
    arrays.values_buffer.selector = action.selector();
    action.finalize();

    auto& keys   = arrays.keys_buffer.selector == 0 ? arrays.keys_in : arrays.keys_out;
    auto& values = arrays.values_buffer.selector == 0 ? arrays.values_in : arrays.values_out;

    arrays.verify_stable_pair_sort(num_items, is_descending, keys, values);
  }
  catch (...)
  {
    c2h::skip_if_allocation_failure();
  }
}

using fp_key_types = c2h::type_list<float, double>;

CUB_TEST("DeviceRadixSort::SortPairs: negative zero handling with bit windows",
         "[pairs][radix][sort][device]",
         CUB_SMALL,
         fp_key_types)
{
  using key_t   = c2h::get<0, TestType>;
  using bits_t  = typename cub::Traits<key_t>::UnsignedBits;
  using value_t = cuda::std::uint32_t;

  constexpr int num_bits    = sizeof(key_t) * CHAR_BIT;
  const key_t positive_zero = cuda::std::bit_cast<key_t>(bits_t(0));
  const key_t negative_zero = cuda::std::bit_cast<key_t>(bits_t(bits_t(1) << (num_bits - 1)));

  // Explicitly use values<>({}) to workaround bug catchorg/Catch2#2040. The sizes are chosen to cover both the
  // single-tile and the multi-pass implementations:
  const std::size_t num_items = GENERATE_COPY(values<std::size_t>({1000, 30000, 300000}));
  const int begin_bit         = GENERATE_COPY(values<int>({0, num_bits / 4, num_bits - 1}));
  const int end_bit           = GENERATE_COPY(values<int>({num_bits / 2, num_bits}));
  if (end_bit <= begin_bit || (begin_bit == 0 && end_bit == num_bits))
  {
    return;
  }

  // Zeros of both signs interleaved with nonzero keys of both signs. The values are the input positions, so the
  // order of equal keys is visible:
  const key_t pattern[] = {positive_zero, key_t(1), key_t(-1), negative_zero, key_t(2), key_t(-2), positive_zero};
  c2h::host_vector<key_t> h_keys(num_items);
  c2h::host_vector<value_t> h_values(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    h_keys[i]   = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    h_values[i] = static_cast<value_t>(i);
  }
  c2h::device_vector<key_t> in_keys(h_keys);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::device_vector<value_t> in_values(h_values);
  c2h::device_vector<value_t> out_values(num_items);

  const bool is_descending = GENERATE(false, true);

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending, begin_bit, end_bit);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      begin_bit,
      end_bit);
  }
  else
  {
    sort_pairs(thrust::raw_pointer_cast(in_keys.data()),
               thrust::raw_pointer_cast(out_keys.data()),
               thrust::raw_pointer_cast(in_values.data()),
               thrust::raw_pointer_cast(out_values.data()),
               num_items,
               begin_bit,
               end_bit);
  }

  // Perform a bitwise comparison to ensure that 0 != -0:
  REQUIRE_BITWISE_EQ(ref_keys, out_keys);
  REQUIRE(ref_values == out_values);
}

CUB_TEST("DeviceRadixSort::SortPairs: 32-bit overflow check",
         "[large][pairs][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck]",
         CUB_LARGE)
{
  using key_t       = std::uint8_t;
  using value_t     = std::uint8_t;
  using num_items_t = std::uint32_t;

  // Test problem size at the maximum offset value to ensure that internal calculations do not overflow.
  const std::size_t num_items = cuda::std::numeric_limits<num_items_t>::max();

  do_large_offset_test<key_t, value_t, num_items_t>(num_items);
}

CUB_TEST("DeviceRadixSort::SortPairs: Large Offsets",
         "[large][pairs][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck]",
         CUB_LARGE)
{
  using key_t       = std::uint8_t;
  using value_t     = std::uint8_t;
  using num_items_t = std::uint64_t;

  constexpr std::size_t min_num_items = std::size_t{1} << 32;
  constexpr std::size_t max_num_items = min_num_items + (std::size_t{1} << 30);
  const std::size_t num_items         = GENERATE_COPY(take(1, random(min_num_items, max_num_items)));

  do_large_offset_test<key_t, value_t, num_items_t>(num_items);
}
