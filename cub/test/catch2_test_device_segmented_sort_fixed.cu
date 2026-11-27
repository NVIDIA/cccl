// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/count.h>

#include <cuda/argument>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1

template <typename KeyT, typename SegmentSizeT, typename NumSegmentsT>
CUB_RUNTIME_FUNCTION cudaError_t sort_fixed_segments(
  void* temp,
  std::size_t& bytes,
  KeyT* keys_in,
  KeyT* keys_out,
  int* values_in,
  int* values_out,
  SegmentSizeT segment_size,
  NumSegmentsT num_segments,
  bool pairs,
  bool descending,
  bool use_double_buffer,
  int initial_selector,
  int* selectors,
  cudaStream_t stream = nullptr)
{
  cub::DoubleBuffer<KeyT> keys(keys_in, keys_out);
  cub::DoubleBuffer<int> values(values_in, values_out);
  keys.selector = values.selector = initial_selector;
  cudaError_t error;
  if (use_double_buffer)
  {
    if (pairs)
    {
      error =
        descending
          ? cub::DeviceSegmentedSort::SortPairsDescending(temp, bytes, keys, values, segment_size, num_segments, stream)
          : cub::DeviceSegmentedSort::SortPairs(temp, bytes, keys, values, segment_size, num_segments, stream);
    }
    else
    {
      error = descending
              ? cub::DeviceSegmentedSort::SortKeysDescending(temp, bytes, keys, segment_size, num_segments, stream)
              : cub::DeviceSegmentedSort::SortKeys(temp, bytes, keys, segment_size, num_segments, stream);
    }
    if (temp != nullptr)
    {
      selectors[0] = keys.selector;
      selectors[1] = values.selector;
    }
  }
  else if (pairs)
  {
    error = descending
            ? cub::DeviceSegmentedSort::SortPairsDescending(
                temp, bytes, keys_in, keys_out, values_in, values_out, segment_size, num_segments, stream)
            : cub::DeviceSegmentedSort::SortPairs(
                temp, bytes, keys_in, keys_out, values_in, values_out, segment_size, num_segments, stream);
  }
  else
  {
    error = descending
            ? cub::DeviceSegmentedSort::SortKeysDescending(
                temp, bytes, keys_in, keys_out, segment_size, num_segments, stream)
            : cub::DeviceSegmentedSort::SortKeys(temp, bytes, keys_in, keys_out, segment_size, num_segments, stream);
  }
  return error;
}

DECLARE_LAUNCH_WRAPPER(sort_fixed_segments, fixed_sort);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortKeys, fixed_sort_keys);

template <typename KeyT, typename SegmentSizeT, typename NumSegmentsT>
void test_fixed_sort(SegmentSizeT segment_size_arg, NumSegmentsT num_segments_arg, int segment_size, int num_segments)
{
  const bool pairs             = GENERATE(false, true);
  const bool descending        = GENERATE(false, true);
  const bool use_double_buffer = GENERATE(false, true);
  const int initial_selector   = use_double_buffer ? GENERATE(0, 1) : 0;
  CAPTURE(segment_size, num_segments, pairs, descending, use_double_buffer, initial_selector);

  const int num_items = segment_size * num_segments;
  c2h::device_vector<KeyT> keys_in(num_items);
  c2h::gen(C2H_SEED(1), keys_in, std::numeric_limits<KeyT>::is_signed ? static_cast<KeyT>(-100) : KeyT{0}, KeyT{100});
  const c2h::host_vector<KeyT> input = keys_in;
  c2h::host_vector<KeyT> expected    = input;
  for (int segment = 0; segment < num_segments; ++segment)
  {
    auto first = expected.begin() + segment * segment_size;
    std::sort(first, first + segment_size, [=](KeyT a, KeyT b) {
      return descending ? a > b : a < b;
    });
  }

  c2h::host_vector<int> input_values(num_items);
  std::iota(input_values.begin(), input_values.end(), 0);
  c2h::device_vector<int> values_in = input_values;
  c2h::device_vector<KeyT> keys_out(num_items);
  c2h::device_vector<int> values_out(num_items);
  int* selectors = nullptr;
  REQUIRE(cudaSuccess == cudaMallocHost(&selectors, 2 * sizeof(int)));
  selectors[0] = selectors[1] = initial_selector;
  auto in_keys                = thrust::raw_pointer_cast(keys_in.data());
  auto out_keys               = thrust::raw_pointer_cast(keys_out.data());
  auto in_vals                = thrust::raw_pointer_cast(values_in.data());
  auto out_vals               = thrust::raw_pointer_cast(values_out.data());
  if (initial_selector == 1)
  {
    std::swap(in_keys, out_keys);
    std::swap(in_vals, out_vals);
  }

  fixed_sort(
    in_keys,
    out_keys,
    in_vals,
    out_vals,
    segment_size_arg,
    num_segments_arg,
    pairs,
    descending,
    use_double_buffer,
    initial_selector,
    selectors);

  const int key_selector   = selectors[0];
  const int value_selector = selectors[1];
  REQUIRE(cudaSuccess == cudaFreeHost(selectors));
  const auto& result_keys = use_double_buffer && key_selector == initial_selector ? keys_in : keys_out;
  REQUIRE(result_keys == expected);
  if (!use_double_buffer)
  {
    REQUIRE(keys_in == input);
    REQUIRE(values_in == input_values);
  }
  if (pairs)
  {
    c2h::host_vector<int> result_values =
      use_double_buffer && value_selector == initial_selector ? values_in : values_out;
    for (int segment = 0; segment < num_segments; ++segment)
    {
      const int begin   = segment * segment_size;
      const int end     = begin + segment_size;
      bool values_match = true;
      for (int i = begin; i < end; ++i)
      {
        values_match = values_match && result_values[i] >= begin && result_values[i] < end
                    && input[result_values[i]] == expected[i];
      }
      REQUIRE(values_match);
      std::sort(result_values.begin() + begin, result_values.begin() + end);
    }
    REQUIRE(result_values == input_values);
  }
}

#if TEST_TYPES == 0
using key_types = c2h::type_list<int, std::uint8_t>;
#else
using key_types = c2h::type_list<float, std::uint64_t>;
#endif

CUB_TEST("DeviceSegmentedSort: fixed-size segments", "[segmented_sort][fixed]", CUB_SMALL, key_types)
{
  using key_t    = c2h::get<0, TestType>;
  const int size = GENERATE(0, 1, 17, 18, 19, 33, 111, 112, 113, 128, 513, 4097, 5887, 5888, 5889, 23551, 23552, 23553);
  const int count = GENERATE(0, 1, 37);
  test_fixed_sort<key_t>(size, std::int64_t{count}, size, count);
}

CUB_TEST("DeviceSegmentedSort: many fixed-size segments", "[segmented_sort][fixed]", CUB_SMALL, key_types)
{
  using key_t    = c2h::get<0, TestType>;
  const int size = GENERATE(1, 33, 129);
  test_fixed_sort<key_t>(size, 2001, size, 2001);
}

CUB_TEST("DeviceSegmentedSort: annotated fixed-size segments", "[segmented_sort][fixed]", CUB_SMALL, key_types)
{
  using key_t = c2h::get<0, TestType>;
  SECTION("compile-time size")
  {
    test_fixed_sort<key_t>(cuda::args::constant<33>{}, cuda::args::immediate{37}, 33, 37);
  }
  SECTION("runtime size with compile-time count")
  {
    const int size = GENERATE(1, 128, 513, 4097);
    test_fixed_sort<key_t>(cuda::args::immediate{size}, cuda::args::constant<37>{}, size, 37);
  }
}

#if TEST_LAUNCH == 0 && TEST_TYPES == 0
CUB_TEST("DeviceSegmentedSort: fixed-size segments beyond a single launch",
         "[segmented_sort][fixed][skip-cs-memcheck][skip-cs-racecheck][skip-cs-initcheck]",
         CUB_LARGE)
try
{
  const auto count = std::int64_t{(std::numeric_limits<int>::max)()} + 32;
  c2h::device_vector<std::uint8_t> input(count, 42);
  c2h::device_vector<std::uint8_t> output(count, 0);
  fixed_sort_keys(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), 1, count);
  REQUIRE(thrust::count(c2h::device_policy, output.begin(), output.end(), std::uint8_t{42}) == count);
}
catch (const std::bad_alloc&)
{
  SKIP("Insufficient device memory for more than INT_MAX fixed-size segments.");
}

CUB_TEST("DeviceSegmentedSort: invalid fixed sizes", "[segmented_sort][fixed]", CUB_SMALL)
{
  int* keys = nullptr;
  size_t bytes{};
  REQUIRE(cudaErrorInvalidValue == cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, keys, keys, -1, 3));
  REQUIRE(cudaErrorInvalidValue == cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, keys, keys, 3, -1));
  REQUIRE(cudaErrorInvalidValue
          == cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, keys, keys, cuda::args::constant<-1>{}, 3));
  REQUIRE(
    cudaErrorInvalidValue
    == cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, keys, keys, (std::numeric_limits<std::uint64_t>::max)(), 1));
  REQUIRE(
    cudaErrorInvalidValue
    == cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, keys, keys, 2, (std::numeric_limits<std::int64_t>::max)()));
}
#endif
