// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_KEY_BITS key_bits 8:16:32:64

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeysDescending, sort_keys_descending);

// TODO:
// - int128
// - uint128

// The unsigned integer for the given byte count should be first:
#if TEST_KEY_BITS == 8
using key_types            = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, bool, char>;
using bit_window_key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, char>;
#  define NO_FP_KEY_TYPES
#  define SINGLE_TEST_CASE_INSTANTIATION
#elif TEST_KEY_BITS == 16
// clang-format off
using key_types = c2h::type_list<
    cuda::std::uint16_t
  , cuda::std::int16_t
#if TEST_HALF_T()
  , half_t
#endif // TEST_HALF_T()
#if TEST_BF_T()
  , bfloat16_t
#endif // TEST_BF_T()
  >;
// clang-format on
using bit_window_key_types = c2h::type_list<cuda::std::uint16_t, cuda::std::int16_t>;
#  define NO_FP_KEY_TYPES
#elif TEST_KEY_BITS == 32
using key_types            = c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t, float>;
using bit_window_key_types = c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t>;
using fp_key_types         = c2h::type_list<float>;
#elif TEST_KEY_BITS == 64
using key_types            = c2h::type_list<cuda::std::uint64_t, cuda::std::int64_t, double>;
using bit_window_key_types = c2h::type_list<cuda::std::uint64_t, cuda::std::int64_t>;
using fp_key_types         = c2h::type_list<double>;
#endif

// Used for tests that just need a single type for testing:
using single_key_type = c2h::type_list<c2h::get<0, key_types>>;

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: basic testing",
         "[keys][segmented][radix][sort][device]",
         key_types,
         offset_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;

  // Use c2h::adjust_seed_count() to reduce cost when running sanitizers:
  const std::size_t num_items = GENERATE_COPY(take(c2h::adjust_seed_count(3), random(min_num_items, max_num_items)));
  const std::size_t num_segments =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, offsets);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
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
    sort_keys(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  REQUIRE((ref_keys == out_keys) == true);
}

#if defined(SINGLE_TEST_CASE_INSTANTIATION)

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: empty data", "[keys][segmented][radix][sort][device]", single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  const std::size_t num_items    = GENERATE(0, take(1, random(0, 1 << 10)));
  const std::size_t num_segments = GENERATE(0, 1);

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<offset_t> offsets(2);
  offsets[0] = 0;
  offsets[1] = 0;

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, offsets);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
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
    sort_keys(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  REQUIRE((ref_keys == out_keys) == true);
}

#endif // defined(SINGLE_TEST_CASE_INSTANTIATION)

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: bit windows",
         "[keys][segmented][radix][sort][device]",
         bit_window_key_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;

  const std::size_t num_items    = GENERATE_COPY(take(c2h::adjust_seed_count(2), random(min_num_items, max_num_items)));
  const std::size_t num_segments = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  constexpr int num_bits = sizeof(key_t) * CHAR_BIT;
  // Explicitly use values<>({}) to workaround bug catchorg/Catch2#2040:
  const int begin_bit = GENERATE_COPY(values<int>({0, num_bits / 3, 3 * num_bits / 4, num_bits}));
  const int end_bit   = GENERATE_COPY(values<int>({0, num_bits / 3, 3 * num_bits / 4, num_bits}));
  if (end_bit < begin_bit || (begin_bit == 0 && end_bit == num_bits))
  {
    // SKIP(); Not available until Catch2 3.3.0
    return;
  }

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, begin_bit, end_bit, is_descending);

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, offsets, begin_bit, end_bit);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit,
      end_bit);
  }
  else
  {
    sort_keys(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit,
      end_bit);
  }

  REQUIRE((ref_keys == out_keys) == true);
}

#if defined(SINGLE_TEST_CASE_INSTANTIATION)

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: large segments", "[keys][segmented][radix][sort][device]", single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 19;
  constexpr std::size_t max_num_items = 1 << 20;

  // Use c2h::adjust_seed_count to reduce cost when running sanitizers:
  const std::size_t num_items    = GENERATE_COPY(take(c2h::adjust_seed_count(2), random(min_num_items, max_num_items)));
  const std::size_t num_segments = 2;

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<offset_t> offsets(3);
  offsets[0] = 0;
  offsets[1] = static_cast<offset_t>(num_items / 2);
  offsets[2] = static_cast<offset_t>(num_items);

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, offsets);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
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
    sort_keys(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  REQUIRE((ref_keys == out_keys) == true);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: DoubleBuffer API",
         "[keys][segmented][radix][sort][device]",
         single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 17;
  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, offsets);

  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));

  double_buffer_segmented_sort_t action(is_descending);
  action.initialize();
  launch(action,
         key_buffer,
         static_cast<int>(num_items),
         static_cast<int>(num_segments),
         // Mix pointers/iterators for segment info to test using different iterable types:
         thrust::raw_pointer_cast(offsets.data()),
         offsets.cbegin() + 1,
         begin_bit<key_t>(),
         end_bit<key_t>());

  key_buffer.selector = action.selector();
  action.finalize();

  auto& keys = key_buffer.selector == 0 ? in_keys : out_keys;

  REQUIRE((ref_keys == keys) == true);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: unspecified ranges",
         "[keys][segmented][radix][sort][device]",
         single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 15;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(num_items / 128, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // will belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

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

  auto ref_keys = segmented_radix_sort_reference(in_keys, is_descending, begin_offsets, end_offsets);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
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
    sort_keys(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(begin_offsets.data()),
      end_offsets.cbegin(),
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  REQUIRE((ref_keys == out_keys) == true);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys: very large num. items and num. segments",
         "[keys][segmented][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
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

  auto offsets = thrust::make_transform_iterator(
    thrust::make_counting_iterator(std::size_t{0}),
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
#  if defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
C2H_TEST("DeviceSegmentedRadixSort::SortKeys: very large segments",
         "[keys][segmented][radix][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
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
#  endif // defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
#endif // defined(SINGLE_TEST_CASE_INSTANTIATION)
