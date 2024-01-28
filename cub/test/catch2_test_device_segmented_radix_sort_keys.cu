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

#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeysDescending, sort_keys_descending);

// %PARAM% TEST_KEY_BITS key_bits 8:16:32:64

// TODO:
// - int128
// - uint128

// The unsigned integer for the given byte count should be first:
#if TEST_KEY_BITS == 8
using key_types            = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, bool, char>;
using bit_window_key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, char>;
#  define NO_FP_KEY_TYPES
#elif TEST_KEY_BITS == 16
// clang-format off
using key_types = c2h::type_list<
    cuda::std::uint16_t
  , cuda::std::int16_t
#ifdef TEST_HALF_T
  , half_t
#endif
#ifdef TEST_BF_T
  , bfloat16_t
#endif
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

// Index types used for OffsetsT testing
using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint64_t>;

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: basic testing",
         "[keys][segmented][radix][sort][device]",
         key_types,
         offset_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items         = GENERATE_COPY(take(3, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(2, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(CUB_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

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

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: empty data", "[keys][segmented][radix][sort][device]", single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  const std::size_t num_items    = GENERATE(0, take(1, random(0, 1 << 10)));
  const std::size_t num_segments = GENERATE(0, 1);

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
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

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: bit windows",
         "[keys][segmented][radix][sort][device]",
         bit_window_key_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items         = GENERATE_COPY(take(2, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(CUB_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

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

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: large segments", "[keys][segmented][radix][sort][device]", single_key_type)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t min_num_items = 1 << 19;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items         = GENERATE_COPY(take(2, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = 2;

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

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

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: DoubleBuffer API",
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
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(CUB_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

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

CUB_TEST("DeviceSegmentedRadixSort::SortKeys: unspecified ranges",
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
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  // Initialize the output keys using the input keys since not all items
  // will belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);

  c2h::device_vector<offset_t> begin_offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(CUB_SEED(num_segment_seeds), begin_offsets, static_cast<offset_t>(num_items));

  // Create separate begin/end offsets arrays and remove some of the segments by
  // setting both offsets to 0.
  c2h::device_vector<offset_t> end_offsets(begin_offsets.cbegin() + 1, begin_offsets.cend());
  begin_offsets.pop_back();

  {
    std::size_t num_empty_segments = num_segments / 16;
    c2h::device_vector<std::size_t> indices(num_empty_segments);
    c2h::gen(CUB_SEED(1), indices, std::size_t{0}, num_segments - 1);
    auto begin = thrust::make_constant_iterator(key_t{0});
    auto end = begin + num_empty_segments;
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
