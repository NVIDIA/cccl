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

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "c2h/custom_type.cuh"
#include "c2h/extended_types.cuh"
#include "catch2/catch.hpp"
#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

DECLARE_CDP_WRAPPER(cub::DeviceScan::ExclusiveSum, device_exclusive_sum);
DECLARE_CDP_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_CDP_WRAPPER(cub::DeviceScan::InclusiveSum, device_inclusive_sum);
DECLARE_CDP_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// %PARAM% TEST_CDP cdp 0:1

// List of types to test
using custom_t = c2h::custom_type_t<c2h::accumulateable_t,
                                    c2h::equal_comparable_t,
                                    c2h::lexicographical_less_comparable_t,
                                    c2h::lexicographical_greater_comparable_t>;

using iterator_type_list =
  c2h::type_list<type_pair<std::int8_t>, type_pair<custom_t>, type_pair<uchar3>>;

CUB_TEST("Device scan works with iterators", "[scan][device]", iterator_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(take(3, random(min_items, max_items)),
                                           values({
                                             min_items,
                                             max_items,
                                           }));

  // Prepare input iterator
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = thrust::make_constant_iterator(default_constant);

  SECTION("inclusive sum")
  {
    using op_t    = cub::Sum;
    using accum_t = cub::detail::accumulator_t<op_t, item_t, item_t>;

    // Prepare verification data
    thrust::host_vector<output_t> expected_result(num_items);
    std::inclusive_scan(in_it, in_it + num_items, expected_result.begin(), op_t{}, accum_t{});

    // Run test
    thrust::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_sum(in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive sum")
  {
    using op_t    = cub::Sum;
    using accum_t = cub::detail::accumulator_t<op_t, item_t, item_t>;

    // Prepare verification data
    thrust::host_vector<output_t> expected_result(num_items);
    std::exclusive_scan(in_it, in_it + num_items, expected_result.begin(), accum_t{}, op_t{});

    // Run test
    thrust::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_sum(in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("inclusive scan")
  {
    using op_t    = cub::Min;
    using accum_t = cub::detail::accumulator_t<op_t, item_t, item_t>;

    // Prepare verification data
    thrust::host_vector<output_t> expected_result(num_items);
    std::inclusive_scan(in_it,
                        in_it + num_items,
                        expected_result.begin(),
                        op_t{},
                        cub::NumericTraits<item_t>::Max());

    // Run test
    thrust::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan(in_it, d_out_it, op_t{}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan")
  {
    using op_t    = cub::Sum;
    using accum_t = cub::detail::accumulator_t<op_t, item_t, item_t>;

    // Prepare verification data
    thrust::host_vector<output_t> expected_result(num_items);
    std::exclusive_scan(in_it, in_it + num_items, expected_result.begin(), accum_t{}, op_t{});

    // Run test
    thrust::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_scan(in_it, d_out_it, op_t{}, item_t{}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan with future-init value")
  {
    using op_t    = cub::Sum;
    using accum_t = cub::detail::accumulator_t<op_t, item_t, item_t>;

    // Prepare verification data
    accum_t init_value{};
    init_default_constant(init_value);
    thrust::host_vector<output_t> expected_result(num_items);
    std::exclusive_scan(in_it, in_it + num_items, expected_result.begin(), init_value, op_t{});

    // Run test
    thrust::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    using init_t  = cub::detail::value_t<decltype(unwrap_it(d_out_it))>;
    thrust::device_vector<init_t> d_initial_value(1);
    d_initial_value[0] = static_cast<init_t>(init_value);
    auto future_init_value =
      cub::FutureValue<init_t>(thrust::raw_pointer_cast(d_initial_value.data()));
    device_exclusive_scan(in_it, d_out_it, op_t{}, future_init_value, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }
}
