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
// above header needs to be included first

#include <cub/device/device_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of offset types to test
using offset_types = c2h::type_list</*std::int32_t, */std::uint32_t, std::uint64_t>;

__host__ __device__ __forceinline__ uint64_t
get_segmented_guassian_sum(const uint64_t num_items, const uint64_t segment_size)
{
  uint64_t sum_per_full_segment = (segment_size * (segment_size - 1)) / 2;
  uint64_t full_segments        = num_items / segment_size;
  uint64_t index_within_segment = num_items % segment_size;

  uint64_t sum_within_partial_segment = (index_within_segment * (index_within_segment - 1)) / 2;
  uint64_t sum_over_full_segments     = full_segments * sum_per_full_segment;
  return sum_within_partial_segment + sum_over_full_segments;
}

template <typename ItemT>
struct mod_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ uint64_t operator()(const uint64_t index) const
  {
    return static_cast<ItemT>(index % segment_size);
  }
};

CUB_TEST("Device reduce works with all device interfaces", "[reduce][device]", offset_types)
{
  using index_t  = uint64_t;
  using offset_t = typename c2h::get<0, TestType>;

  CAPTURE(c2h::type_name<offset_t>());

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_max_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items_max = static_cast<offset_t>(num_items_max_ull);
  offset_t num_items_min =
    num_items_max_ull > 10000 ? static_cast<offset_t>(num_items_max_ull - 10000ULL) : offset_t{0};

  // Generate the input sizes to test for
  offset_t num_items = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  // Input data
  auto index_it = thrust::make_counting_iterator(index_t{});

  SECTION("reduce")
  {
    using op_t = cub::Sum;

    // Segment size (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
    const auto segment_size = 1000;

    // Initial value of reduction
    const auto init_val = index_t{42};

    // Binary reduction operator
    auto reduction_op = op_t{};

    // Prepare verification data
    index_t expected_result = init_val + get_segmented_guassian_sum(num_items, segment_size);

    // Run test
    c2h::device_vector<index_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_in_it  = thrust::make_transform_iterator(index_it, mod_op<index_t>{segment_size});

    device_reduce(d_in_it, d_out_it, num_items, reduction_op, init_val);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("sum")
  {
    // Segment size (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
    const auto segment_size = 1000;

    // Prepare verification data
    index_t expected_result = get_segmented_guassian_sum(num_items, segment_size);

    // Run test
    c2h::device_vector<index_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_in_it  = thrust::make_transform_iterator(index_it, mod_op<index_t>{segment_size});

    device_sum(d_in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("min")
  {
    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<index_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_in_it  = thrust::make_reverse_iterator(index_it + num_items + iterator_offset);

    device_min(d_in_it, d_out_it, num_items);

    // Verify result
    const index_t expected_result = iterator_offset;
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("max")
  {
    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<index_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_in_it  = index_it + iterator_offset;

    device_max(d_in_it, d_out_it, num_items);

    // Verify result
    const index_t expected_result = num_items + iterator_offset - index_t{1};
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("argmin")
  {
    using result_t = cub::KeyValuePair<offset_t, index_t>;

    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<result_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    auto d_in_it  = thrust::make_reverse_iterator(index_it + num_items + iterator_offset);

    device_arg_min(d_in_it, &d_out_it->value, &d_out_it->key, num_items);

    // Verify result
    const index_t expected_value = iterator_offset;
    const index_t expected_index = num_items - 1;

    // Verify result
    result_t gpu_result = out_result[0];
    REQUIRE(expected_value == gpu_result.value);
    REQUIRE(expected_index == gpu_result.key);
  }

  // SECTION("argmax int-key-value-pair")
  // {
  //   using result_t = cub::KeyValuePair<offset_t, index_t>;

  //   // Run test
  //   const index_t iterator_offset = 1000;
  //   c2h::device_vector<result_t> out_result(1);
  //   auto d_out_it = thrust::raw_pointer_cast(out_result.data());
  //   auto d_in_it  = index_it + iterator_offset;

  //   device_arg_max(d_in_it, d_out_it, num_items);

  //   // Verify result
  //   const index_t expected_value = num_items + iterator_offset - index_t{1};
  //   const index_t expected_index = num_items + - index_t{1};

  //   // Verify result
  //   result_t gpu_result = out_result[0];
  //   REQUIRE(expected_value == gpu_result.value);
  //   REQUIRE(expected_index == gpu_result.key);
  // }

  // Argmax still works with an output iterator that has KeyValuePair<int, T> value type
  // SECTION("argmax int-kv pair")
  // {
  //   // Prepare verification data
  //   c2h::host_vector<item_t> host_items(in_items);
  //   auto expected_result = std::max_element(host_items.cbegin(), host_items.cend());

  //   // Run test

  //   using result_t = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
  //   c2h::device_vector<result_t> out_result(num_segments);
  //   device_arg_max(unwrap_it(d_in_it), thrust::raw_pointer_cast(out_result.data()), num_items);

  //   // Verify result
  //   result_t gpu_result = out_result[0];
  //   output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
  //   REQUIRE(expected_result[0] == gpu_value);
  //   REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  // }

#if false
  SECTION("argmin int")
  {
    // Prepare verification data
    c2h::host_vector<item_t> host_items(in_items);
    auto expected_result = std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
    c2h::device_vector<result_t> out_result(num_segments);
    device_arg_min(unwrap_it(d_in_it), thrust::raw_pointer_cast(out_result.data()), num_items);

    // Verify result
    result_t gpu_result = out_result[0];
    output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_value);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  }
#endif
}
