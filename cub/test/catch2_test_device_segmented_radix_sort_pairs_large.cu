// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceSegmentedRadixSort::SortPairs tests, split out from
// catch2_test_device_segmented_radix_sort_pairs.cu so they can run serially (tagged [large-mem])
// without serializing the small tests in that file. All helpers come from the shared headers.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/memory.h>
#include <thrust/scatter.h>

#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new> // bad_alloc

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairsDescending, sort_pairs_descending);

C2H_TEST("DeviceSegmentedSortPairs: very large num. items and num. segments",
         "[large-mem][pairs][segmented][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         all_offset_types)
try
{
  using key_t                        = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using value_t                      = cuda::std::uint8_t;
  using segment_offset_t             = std::int64_t;
  using offset_t                     = c2h::get<0, TestType>;
  using segment_iterator_t           = segment_index_to_offset_op<offset_t, segment_offset_t>;
  constexpr std::size_t segment_size = 1000000;
  constexpr std::size_t uint32_max   = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : cuda::std::numeric_limits<offset_t>::max();
  constexpr segment_offset_t num_empty_segments = uint32_max;
  const segment_offset_t num_segments           = num_empty_segments + cuda::ceil_div(num_items, segment_size);
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  // Generate input
  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<value_t> in_values(num_items);
  constexpr auto max_histo_size = 250;
  segmented_verification_helper<key_t> verification_helper{max_histo_size};
  verification_helper.prepare_input_data(in_keys);
  thrust::copy(in_keys.cbegin(), in_keys.cend(), in_values.begin());

  // Initialize the output vectors by copying the inputs since not all items may belong to a segment.
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::device_vector<value_t> out_values(num_items);

  auto offsets = cuda::transform_iterator(
    cuda::counting_iterator(std::size_t{0}),
    segment_iterator_t{num_empty_segments, num_segments, segment_size, num_items});
  auto offsets_plus_1 = offsets + 1;

  sort_pairs(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    thrust::raw_pointer_cast(in_values.data()),
    thrust::raw_pointer_cast(out_values.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    offsets,
    offsets_plus_1,
    begin_bit<key_t>(),
    end_bit<key_t>());

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys, offsets + num_empty_segments, num_segments - num_empty_segments);

  // Verify values were sorted along with the keys
  REQUIRE(thrust::equal(out_keys.cbegin(), out_keys.cend(), out_values.cbegin()));
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented sort test, insufficient GPU memory. " << e.what() << "\n";
}

// Currently, size of a single segment in DeviceRadixSort is limited to INT_MAX
#if defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
C2H_TEST("DeviceSegmentedSort::SortPairs: very large segments",
         "[large-mem][pairs][segmented][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         all_offset_types)
try
{
  using key_t                      = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using value_t                    = cuda::std::uint8_t;
  using segment_offset_t           = std::int32_t;
  using offset_t                   = c2h::get<0, TestType>;
  constexpr std::size_t uint32_max = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr int num_key_seeds      = 1;
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : cuda::std::numeric_limits<offset_t>::max();
  constexpr segment_offset_t num_segments = 2;
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  thrust::copy(in_keys.cbegin(), in_keys.cend(), in_values.begin());
  c2h::device_vector<value_t> out_values(num_items);
  c2h::device_vector<offset_t> offsets(num_segments + 1);
  offsets[0] = 0;
  offsets[1] = static_cast<offset_t>(num_items);
  offsets[2] = static_cast<offset_t>(num_items);

  // Prepare information for later verification
  short_key_verification_helper<key_t> verification_helper{};
  verification_helper.prepare_verification_data(in_keys);

  sort_pairs(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    thrust::raw_pointer_cast(in_values.data()),
    thrust::raw_pointer_cast(out_values.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    thrust::raw_pointer_cast(offsets.data()),
    offsets.cbegin() + 1,
    begin_bit<key_t>(),
    end_bit<key_t>());

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys);

  // Verify values were sorted along with the keys
  REQUIRE(thrust::equal(out_keys.cbegin(), out_keys.cend(), out_values.cbegin()));
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented sort test, insufficient GPU memory. " << e.what() << "\n";
}
#endif // defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
