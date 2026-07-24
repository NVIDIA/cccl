// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceSegmentedRadixSort::SortKeys tests, split out from
// catch2_test_device_segmented_radix_sort_keys.cu so they can run serially (tagged [large-mem])
// without serializing the small tests. The tests hardcode uint8 keys, so TEST_KEY_BITS is not
// needed here (the original gated them to a single key-bits instantiation for the same reason).

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/functional.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new> // bad_alloc

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeysDescending, sort_keys_descending);

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: very large num. items and num. segments",
         "[large-mem][keys][segmented][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         all_offset_types)
try
{
  using key_t                        = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using segment_offset_t             = std::int64_t;
  using offset_t                     = c2h::get<0, TestType>;
  using segment_iterator_t           = segment_index_to_offset_op<offset_t, segment_offset_t>;
  constexpr std::size_t segment_size = 1000000;
  constexpr std::size_t uint32_max   = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 22) : cuda::std::numeric_limits<offset_t>::max();
  constexpr segment_offset_t num_empty_segments = uint32_max - 5U;
  const segment_offset_t num_segments           = num_empty_segments + cuda::ceil_div(num_items, segment_size);
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  // Generate input keys
  constexpr auto max_histo_size = 250;
  segmented_verification_helper<key_t> verification_helper{max_histo_size};
  verification_helper.prepare_input_data(in_keys);

  auto offsets = cuda::transform_iterator(
    cuda::counting_iterator(std::size_t{0}),
    segment_iterator_t{num_empty_segments, num_segments, segment_size, num_items});

  sort_keys(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    offsets,
    offsets + 1,
    begin_bit<key_t>(),
    end_bit<key_t>());

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys, offsets + num_empty_segments, num_segments - num_empty_segments);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented radix sort test, insufficient GPU memory. " << e.what() << "\n";
}

// Currently, size of a single segment in DeviceRadixSort is limited to INT_MAX
#if defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
C2H_TEST("DeviceSegmentedRadixSort::SortKeys: very large segments",
         "[large-mem][keys][segmented][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         all_offset_types)
try
{
  using key_t                      = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using segment_offset_t           = std::int32_t;
  using offset_t                   = c2h::get<0, TestType>;
  constexpr std::size_t uint32_max = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr int num_key_seeds      = 1;
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : cuda::std::numeric_limits<offset_t>::max();
  const segment_offset_t num_segments = 2;
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::device_vector<offset_t> offsets(num_segments + 1);
  offsets[0] = 0;
  offsets[1] = static_cast<offset_t>(num_items);
  offsets[2] = static_cast<offset_t>(num_items);

  // Prepare information for later verification
  short_key_verification_helper<key_t> verification_helper{};
  verification_helper.prepare_verification_data(in_keys);

  sort_keys(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    thrust::raw_pointer_cast(offsets.data()),
    offsets.cbegin() + 1,
    begin_bit<key_t>(),
    end_bit<key_t>());

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented radix sort test, insufficient GPU memory. " << e.what() << "\n";
}
#endif // defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
