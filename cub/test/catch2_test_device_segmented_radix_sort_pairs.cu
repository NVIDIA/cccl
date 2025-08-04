/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairsDescending, sort_pairs_descending);

using custom_value_t = c2h::custom_type_t<c2h::equal_comparable_t>;
using value_types    = c2h::type_list<cuda::std::uint8_t, cuda::std::uint64_t, custom_value_t>;

// Index types used for OffsetsT testing
C2H_TEST("DeviceSegmentedRadixSort::SortPairs: Basic testing",
         "[pairs][segmented][radix][sort][device]",
         value_types,
         offset_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;

  // Use c2h::adjust_seed_count to reduce runtime with sanitizers:
  const std::size_t num_items = GENERATE_COPY(take(c2h::adjust_seed_count(3), random(min_num_items, max_num_items)));
  const std::size_t num_segments =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: DoubleBuffer API", "[pairs][segmented][radix][sort][device]", value_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  cub::DoubleBuffer<value_t> value_buffer(
    thrust::raw_pointer_cast(in_values.data()), thrust::raw_pointer_cast(out_values.data()));

  double_buffer_segmented_sort_t action(is_descending);
  action.initialize();
  launch(action,
         key_buffer,
         value_buffer,
         static_cast<int>(num_items),
         static_cast<int>(num_segments),
         // Mix pointers/iterators for segment info to test using different iterable types:
         thrust::raw_pointer_cast(offsets.data()),
         offsets.cbegin() + 1,
         begin_bit<key_t>(),
         end_bit<key_t>());

  key_buffer.selector   = action.selector();
  value_buffer.selector = action.selector();
  action.finalize();

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  auto& keys   = key_buffer.selector == 0 ? in_keys : out_keys;
  auto& values = value_buffer.selector == 0 ? in_values : out_values;

  REQUIRE(ref_keys == keys);
  REQUIRE(ref_values == values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: unspecified ranges",
         "[pairs][segmented][radix][sort][device]",
         value_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  c2h::device_vector<offset_t> begin_offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), begin_offsets, static_cast<offset_t>(num_items));

  // Create separate begin/end offsets arrays and remove some of the segments by
  // setting both offsets to 0.
  c2h::device_vector<offset_t> end_offsets(begin_offsets.cbegin() + 1, begin_offsets.cend());
  begin_offsets.pop_back();

  {
    std::size_t num_empty_segments = num_segments / 16;
    c2h::device_vector<std::size_t> indices(num_empty_segments);
    c2h::gen(C2H_SEED(1), indices, std::size_t{0}, num_segments - 1);
    auto begin = thrust::make_constant_iterator(key_t{0});
    auto end   = begin + num_empty_segments;
    thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), begin_offsets.begin());
    thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), end_offsets.begin());
  }

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(begin_offsets.data()),
      end_offsets.cbegin(),
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(begin_offsets.data()),
      end_offsets.cbegin(),
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, begin_offsets, end_offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE((ref_keys == out_keys) == true);
  REQUIRE((ref_values == out_values) == true);
}

C2H_TEST("DeviceSegmentedSortPairs: very large num. items and num. segments",
         "[pairs][segmented][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
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

  auto offsets = thrust::make_transform_iterator(
    thrust::make_counting_iterator(std::size_t{0}),
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
         "[pairs][segmented][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
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
