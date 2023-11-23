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

// Catch2 internals throw the following diagnositic when CUB_TEST is given an empty typelist,
// which happens for some parameterizations of this test due to filtering for floating-point types, etc.
// error #177-D: variable "index" was declared but never referenced
#pragma diag_suppress 177

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeysDescending, sort_keys_descending);

// %PARAM% TEST_KEY_BITS key_bits 8:16:32:64

// TODO:
// - int128
// - uint128

// The unsigned integer for the given byte count should be first:
#if TEST_KEY_BITS == 8
using key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, bool, char>;
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
#elif TEST_KEY_BITS == 32
using key_types = c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t, float>;
#elif TEST_KEY_BITS == 64
using key_types = c2h::type_list<cuda::std::uint64_t, cuda::std::int64_t, double>;
#endif

// Used for testing begin/end bit slicing
template <typename T>
struct supports_bit_window
{
  static constexpr bool value = cuda::std::is_integral<T>::value && !cuda::std::is_same<bool, T>::value;
};
using bit_window_key_types = c2h::copy_if<key_types, c2h::trait<supports_bit_window>>;

// Used for testing floating-point specific behavior
using fp_key_types = c2h::copy_if<key_types, c2h::trait<cuda::std::is_floating_point>>;

// Used for tests that just need a single type for testing:
using single_key_type = c2h::type_list<c2h::get<0, key_types>>;

// Index types used for NumItemsT testing
using num_items_types =
  c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t, cuda::std::uint64_t, cuda::std::int64_t>;

CUB_TEST("DeviceRadixSort::SortKeys: basic testing", "[keys][radix][sort][device]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items = GENERATE_COPY(std::size_t{0}, std::size_t{1}, take(8, random(min_num_items, max_num_items)));

  thrust::device_vector<key_t> in_keys(num_items);
  thrust::device_vector<key_t> out_keys(num_items);

  const int num_key_seeds = 3;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  REQUIRE(ref_keys == out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: bit windows", "[keys][radix][sort][device]", bit_window_key_types)
{
  using key_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));

  constexpr int num_bits = sizeof(key_t) * CHAR_BIT;
  const int begin_bit    = GENERATE_COPY(0, num_bits / 3, 3 * num_bits / 4, num_bits);
  const int end_bit      = GENERATE_COPY(0, num_bits / 3, 3 * num_bits / 4, num_bits);
  if (end_bit < begin_bit || (begin_bit == 0 && end_bit == num_bits))
  {
    // SKIP(); Not available until Catch2 3.3.0
    return;
  }

  thrust::device_vector<key_t> in_keys(num_items);
  thrust::device_vector<key_t> out_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending, begin_bit, end_bit);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit,
      end_bit);
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit,
              end_bit);
  }

  REQUIRE(ref_keys == out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: negative zero handling", "[keys][radix][sort][device]", fp_key_types)
{
  using key_t  = c2h::get<0, TestType>;
  using bits_t = typename cub::Traits<key_t>::UnsignedBits;

  constexpr std::size_t num_bits = sizeof(key_t) * CHAR_BIT;
  const key_t positive_zero      = c2h::bit_cast<key_t>(bits_t(0));
  const key_t negative_zero      = c2h::bit_cast<key_t>(bits_t(1) << (num_bits - 1));

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  thrust::device_vector<key_t> in_keys(num_items);
  thrust::device_vector<key_t> out_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  // Sprinkle some positive and negative zeros randomly throughout the keys:
  {
    const size_t num_indices = num_items / 128;
    thrust::device_vector<std::size_t> indices(num_indices);
    for (int i = 0; i < 2; ++i)
    {
      c2h::gen(CUB_SEED(1), indices, std::size_t(0), num_items);
      auto begin = thrust::make_constant_iterator(i == 0 ? positive_zero : negative_zero);
      auto end   = begin + num_indices;
      thrust::scatter(begin, end, indices.cbegin(), in_keys.begin());
    }
  }

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  // Perform a bitwise comparison to ensure that 0 != -0:
  REQUIRE_BITWISE_EQ(ref_keys, out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: NaN handling", "[keys][radix][sort][device]", fp_key_types)
{
  using key_t    = c2h::get<0, TestType>;
  using limits_t = cuda::std::numeric_limits<key_t>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  thrust::device_vector<key_t> in_keys(num_items);
  thrust::device_vector<key_t> out_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  // Sprinkle some NaNs randomly throughout the keys:
  {
    const size_t num_indices = num_items / 128;
    thrust::device_vector<std::size_t> indices(num_indices);
    bool has_nans = false;
    for (int i = 0; i < 2; ++i)
    {
      const bool supported = i == 0 ? limits_t::has_signaling_NaN : limits_t::has_quiet_NaN;
      const key_t nan_val  = i == 0 ? limits_t::signaling_NaN() : limits_t::quiet_NaN();

      if (supported)
      {
        has_nans = true;
        c2h::gen(CUB_SEED(1), indices, std::size_t(0), num_items);
        auto begin = thrust::make_constant_iterator(nan_val);
        auto end   = begin + num_indices;
        thrust::scatter(begin, end, indices.cbegin(), in_keys.begin());
      }
    }
    if (!has_nans)
    {
      // SKIP(); Not available until Catch2 3.3.0
      return;
    }
  }

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  REQUIRE_EQ_WITH_NAN_MATCHING(ref_keys, out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: entropy reduction", "[keys][radix][sort][device]", single_key_type)
{
  using key_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  thrust::device_vector<key_t> in_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  // Repeatedly bitwise-and random keys together. This increases the likelyhood
  // of duplicate keys.
  const int entropy_reduction = GENERATE(1, 3, 9, 15);
  {
    thrust::device_vector<key_t> tmp(num_items);
    for (int i = 0; i < entropy_reduction; ++i)
    {
      c2h::gen(CUB_SEED(1), tmp);
      thrust::transform(in_keys.cbegin(), in_keys.cend(), tmp.cbegin(), in_keys.begin(), thrust::bit_and<key_t>{});
    }
  }

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  thrust::device_vector<key_t> out_keys(num_items);
  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  REQUIRE(ref_keys == out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: uniform values", "[keys][radix][sort][device]", key_types)
{
  using key_t  = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  thrust::device_vector<key_t> in_keys(num_items, key_t(4));

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  thrust::device_vector<key_t> out_keys(num_items);
  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  REQUIRE(ref_keys == out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: NumItemsT", "[keys][radix][sort][device]", single_key_type, num_items_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  constexpr num_items_t min_num_items = 1 << 5;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(8, random(min_num_items, max_num_items)));

  thrust::device_vector<key_t> in_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  thrust::device_vector<key_t> out_keys(num_items);
  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              begin_bit<key_t>(),
              end_bit<key_t>());
  }

  REQUIRE(ref_keys == out_keys);
}

CUB_TEST("DeviceRadixSort::SortKeys: DoubleBuffer API", "[keys][radix][sort][device]", single_key_type)
{
  using key_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  thrust::device_vector<key_t> in_keys(num_items);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  const bool is_descending = GENERATE(false, true);

  auto ref_keys = radix_sort_reference(in_keys, is_descending);

  thrust::device_vector<key_t> out_keys(num_items);
  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action,
         key_buffer,
         num_items,
         begin_bit<key_t>(),
         end_bit<key_t>());

  key_buffer.selector = action.selector();
  action.finalize();

  auto& keys = key_buffer.selector == 0 ? in_keys : out_keys;

  REQUIRE(ref_keys == keys);
}
