// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <cuda/buffer>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cstddef>
#include <cstdint>

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_allocation_failure.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/cpu_timer.h>
#include <c2h/custom_type.h>
#include <c2h/device_and_stream.h>

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
  const auto stream = c2h::make_current_device_stream();

  using key_t       = cuda::std::uint32_t;
  using value_t     = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  constexpr num_items_t min_num_items = 1 << 5;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(5, random(min_num_items, max_num_items)));

  auto key_buffers   = c2h::gen_buffers<key_t>(stream, C2H_SEED(1), num_items);
  auto value_buffers = c2h::gen_buffers<value_t>(stream, C2H_SEED(1), num_items);
  auto& in_keys      = key_buffers.d_items;
  auto& in_values    = value_buffers.d_items;
  auto out_keys      = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  auto out_values    = c2h::make_device_buffer<value_t>(stream, num_items, cuda::no_init);

  const bool is_descending = GENERATE(false, true);
  const auto reference     = radix_sort_reference(stream, key_buffers.h_items, value_buffers.h_items, is_descending);

  if (is_descending)
  {
    sort_pairs_descending(
      in_keys.data(),
      out_keys.data(),
      in_values.data(),
      out_values.data(),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      stream);
  }
  else
  {
    sort_pairs(
      in_keys.data(),
      out_keys.data(),
      in_values.data(),
      out_values.data(),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      stream);
  }

  REQUIRE_THAT_QUIET(out_keys, c2h::Equals(stream, reference.first));
  REQUIRE_THAT_QUIET(out_values, c2h::Equals(stream, reference.second));
}

CUB_TEST("DeviceRadixSort::SortPairs: DoubleBuffer API", "[pairs][radix][sort][device]", CUB_SMALL, value_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t   = cuda::std::uint32_t;
  using value_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));

  auto key_buffers   = c2h::gen_buffers<key_t>(stream, C2H_SEED(1), num_items);
  auto value_buffers = c2h::gen_buffers<value_t>(stream, C2H_SEED(1), num_items);
  auto& in_keys      = key_buffers.d_items;
  auto& in_values    = value_buffers.d_items;
  auto out_keys      = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  auto out_values    = c2h::make_device_buffer<value_t>(stream, num_items, cuda::no_init);

  const bool is_descending = GENERATE(false, true);
  const auto reference     = radix_sort_reference(stream, key_buffers.h_items, value_buffers.h_items, is_descending);

  cub::DoubleBuffer<key_t> key_buffer(in_keys.data(), out_keys.data());
  cub::DoubleBuffer<value_t> value_buffer(in_values.data(), out_values.data());

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, key_buffer, value_buffer, num_items, begin_bit<key_t>(), end_bit<key_t>(), stream);

  key_buffer.selector   = action.selector();
  value_buffer.selector = action.selector();
  action.finalize();

  auto& keys   = key_buffer.selector == 0 ? in_keys : out_keys;
  auto& values = value_buffer.selector == 0 ? in_values : out_values;

  REQUIRE_THAT_QUIET(keys, c2h::Equals(stream, reference.first));
  REQUIRE_THAT_QUIET(values, c2h::Equals(stream, reference.second));
}

template <typename KeyT, typename ValueT, typename NumItemsT>
void do_large_offset_test(std::size_t num_items)
{
  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, is_descending);

  try
  {
    const auto stream = c2h::make_current_device_stream();
    large_array_sort_helper<KeyT, ValueT> arrays(stream);
    arrays.initialize_for_stable_pair_sort(C2H_SEED(1), num_items, is_descending);

    TIME(c2h::cpu_timer timer);

    double_buffer_sort_t action(is_descending);
    action.initialize();
    const NumItemsT typed_num_items = static_cast<NumItemsT>(num_items);
    launch(
      action, arrays.keys_buffer, arrays.values_buffer, typed_num_items, begin_bit<KeyT>(), end_bit<KeyT>(), stream);

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
