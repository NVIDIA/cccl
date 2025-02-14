// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>

#include <thrust/fill.h>

#include <c2h/catch2_test_helper.h>

template <std::uint32_t NumItems,
          std::uint32_t MaxItemValue,
          cub::detail::batch_memcpy::prefer_power_of_two_bits_option PreferPowerOfTwoBits>
__global__ void test_bit_packed_counter_kernel(
  std::uint32_t* bins, std::uint32_t* increments, std::uint32_t* counts_out, std::int32_t num_items)
{
  using bit_packed_counter_t =
    cub::detail::batch_memcpy::bit_packed_counter<NumItems, MaxItemValue, PreferPowerOfTwoBits>;
  bit_packed_counter_t counter{};
  for (std::int32_t i = 0; i < num_items; i++)
  {
    counter.add(bins[i], increments[i]);
  }

  for (std::uint32_t i = 0; i < NumItems; i++)
  {
    counts_out[i] = counter.get(i);
  }
}

template <std::uint32_t NumItems, std::uint32_t MaxItemValue>
struct test_spec
{
  static constexpr std::uint32_t num_items      = NumItems;
  static constexpr std::uint32_t max_item_value = MaxItemValue;
};

using use_power_of_two_bits =
  c2h::enum_type_list<cub::detail::batch_memcpy::prefer_power_of_two_bits_option,
                      cub::detail::batch_memcpy::prefer_power_of_two_bits_option::yes,
                      cub::detail::batch_memcpy::prefer_power_of_two_bits_option::no>;

using test_combinations =
  c2h::type_list<test_spec<1, 1>,
                 test_spec<1, (0x01U << 16)>,
                 test_spec<4, 1>,
                 test_spec<4, 2>,
                 test_spec<4, 255>,
                 test_spec<4, 256>,
                 test_spec<8, 1024>,
                 test_spec<32, 1>,
                 test_spec<32, 256>>;

C2H_TEST("The bit_packed_counter used by DeviceMemcpy works", "[memcpy]", test_combinations, use_power_of_two_bits)
{
  constexpr std::uint32_t num_items       = c2h::get<0, TestType>::num_items;
  constexpr std::uint32_t max_item_value  = c2h::get<0, TestType>::max_item_value;
  constexpr auto prefer_power_of_two_bits = c2h::get<1, TestType>::value;

  constexpr std::uint32_t min_increment = 0;
  constexpr std::uint32_t max_increment = 4;
  constexpr double avg_increment =
    static_cast<double>(min_increment) + (static_cast<double>(max_increment - min_increment) / 2.0);
  auto const num_increments =
    static_cast<std::int32_t>(static_cast<double>(max_item_value * num_items) / avg_increment);
  CAPTURE(num_items, max_item_value, prefer_power_of_two_bits, num_increments);

  // Initialize device-side test data
  c2h::device_vector<std::uint32_t> bins_in(num_increments);
  c2h::gen(C2H_SEED(2), bins_in, 0U, num_items - 1U);
  c2h::device_vector<std::uint32_t> increments_in(num_increments);
  c2h::gen(C2H_SEED(2), increments_in, min_increment, max_increment);

  c2h::device_vector<std::uint32_t> h_bins(bins_in);
  c2h::device_vector<std::uint32_t> h_increments(increments_in);
  c2h::host_vector<std::uint32_t> reference_counters(num_items, 0);

  // Make sure test data does not overflow any of the counters
  for (std::int32_t i = 0; i < num_increments; i++)
  {
    // New increment for this bin would overflow => zero this increment
    if (reference_counters[h_bins[i]] + h_increments[i] >= max_item_value)
    {
      h_increments[i] = 0;
    }
    else
    {
      reference_counters[h_bins[i]] += h_increments[i];
    }
  }

  // Flush back the modified increments
  increments_in = h_increments;

  // Prepare verification data
  c2h::device_vector<std::uint32_t> counts_out(num_items, 814920U);

  // Run tests with densely bit-packed counters
  test_bit_packed_counter_kernel<num_items, max_item_value, prefer_power_of_two_bits><<<1, 1>>>(
    thrust::raw_pointer_cast(bins_in.data()),
    thrust::raw_pointer_cast(increments_in.data()),
    thrust::raw_pointer_cast(counts_out.data()),
    num_increments);

  // Result verification
  REQUIRE(reference_counters == counts_out);
}
