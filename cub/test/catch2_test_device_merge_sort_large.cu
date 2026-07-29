// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceMergeSort test, split out from catch2_test_device_merge_sort.cu so it can be
// run serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>

#include <cuda/iterator>
#include <cuda/std/iterator>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new> // bad_alloc

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);

template <typename OffsetT, typename KeyT = std::uint8_t>
struct type_tuple
{
  using offset_t = OffsetT;
  using key_t    = KeyT;
};

using offset_types =
  c2h::type_list<type_tuple<std::int16_t>,
                 type_tuple<std::int32_t, std::uint32_t>,
                 type_tuple<std::uint32_t>,
                 type_tuple<std::uint64_t>>;

// In combination with a counting iterator, generates a sequence that wraps around after reaching
// `UnsignedIntegralKeyT`'s maximum value.
template <typename UnsignedIntegralKeyT>
struct index_to_key_value_op
{
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    return static_cast<UnsignedIntegralKeyT>(index % num_distinct_key_values);
  }
};

// In combination with a counting iterator, generates the expected sorted order for a sequence
// generated with `index_to_key_value_op`.
template <typename UnsignedIntegralKeyT>
class index_to_expected_key_op
{
private:
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  std::size_t expected_count_per_item;
  std::size_t num_remainder_items;
  std::size_t remainder_item_count;

public:
  index_to_expected_key_op(std::size_t num_total_items)
      : expected_count_per_item(num_total_items / num_distinct_key_values)
      , num_remainder_items(num_total_items % num_distinct_key_values)
      , remainder_item_count(expected_count_per_item + std::size_t{1ULL})
  {}

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    std::size_t remainder_items_offset = num_remainder_items * remainder_item_count;

    UnsignedIntegralKeyT target_item_index =
      (index <= remainder_items_offset)
        ? static_cast<UnsignedIntegralKeyT>(index / remainder_item_count)
        : static_cast<UnsignedIntegralKeyT>(
            num_remainder_items + ((index - remainder_items_offset) / expected_count_per_item));
    return target_item_index;
  }
};

C2H_TEST("DeviceMergeSort::StableSortPairs works for large inputs",
         "[large-mem][merge][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
{
  using testing_types_tuple = c2h::get<0, TestType>;
  using key_t               = typename testing_types_tuple::key_t;
  using offset_t            = typename testing_types_tuple::offset_t;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_ull = std::min(static_cast<std::size_t>(cuda::std::numeric_limits<offset_t>::max()) - 1,
                                cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items = static_cast<offset_t>(num_items_ull);

  SECTION("Random")
  {
    try
    {
      large_array_sort_helper<key_t> arrays;
      constexpr bool is_descending = false;
      arrays.initialize_for_unstable_key_sort(C2H_SEED(1), num_items, is_descending);

      arrays.deallocate_outputs();

      stable_sort_keys(thrust::raw_pointer_cast(arrays.keys_in.data()), num_items, custom_less_op_t{});

      arrays.verify_unstable_key_sort(num_items, is_descending, arrays.keys_in);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }

  SECTION("Pre-sorted input")
  {
    try
    {
      c2h::device_vector<key_t> keys_in_out(num_items);

      auto counting_it = cuda::counting_iterator(std::size_t{0});
      thrust::copy(counting_it, counting_it + num_items, keys_in_out.begin());

      stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

      auto expected_result_it =
        cuda::transform_iterator(cuda::counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      bool is_correct = thrust::equal(expected_result_it, expected_result_it + num_items, keys_in_out.begin());
      REQUIRE(is_correct == true);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }

  SECTION("Reverse-sorted input")
  {
    try
    {
      c2h::device_vector<key_t> keys_in_out(num_items);

      auto counting_it   = cuda::counting_iterator(std::size_t{0});
      auto key_value_it  = cuda::transform_iterator(counting_it, index_to_key_value_op<key_t>{});
      auto rev_sorted_it = cuda::std::make_reverse_iterator(key_value_it + num_items);
      thrust::copy(rev_sorted_it, rev_sorted_it + num_items, keys_in_out.begin());

      stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

      auto expected_result_it =
        cuda::transform_iterator(cuda::counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      bool is_correct = thrust::equal(expected_result_it, expected_result_it + num_items, keys_in_out.cbegin());
      REQUIRE(is_correct == true);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }
}
