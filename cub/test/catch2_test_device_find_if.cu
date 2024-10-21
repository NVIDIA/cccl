/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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

#include <cub/device/device_find_if.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include <iostream>

#include "c2h/custom_type.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <catch2/catch.hpp>
#include <nv/target>

// %PARAM% TEST_LAUNCH lid 0:1

// DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, device_findif);

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

using full_type_list = c2h::type_list<type_pair<std::uint8_t, std::int32_t>, type_pair<std::int8_t>>;
// clang-format on

enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

template <typename InputIt, typename OutputIt, typename BinaryOp>
void compute_find_if_reference(InputIt first, InputIt last, OutputIt& result, BinaryOp op)
{
  auto pos = thrust::find_if(first, last, op);
  result   = pos - first;
}

template <typename T>
struct equals
{
  T val;
  equals(T _val)
      : val(_val)
  {}

  __device__ __host__ bool operator()(T i)
  {
    return i == val;
  }
};

CUB_TEST("Device find_if works", "[device]", full_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
  {
    c2h::gen(CUB_SEED(2), in_items);
  }
  else
  {
    input_t default_constant{};
    init_default_constant(default_constant);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  SECTION("Generic find if case")
  {
    using op_t = equals<input_t>;
    input_t val_to_find{2};

    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(host_items.begin(), host_items.end(), expected_result[0], op_t{val_to_find});

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes{};

    // Run test
    c2h::device_vector<output_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{val_to_find}, num_items);

    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{val_to_find}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("find_if works with non raw pointers - .begin() iterator")
  {
    using op_t = equals<input_t>;
    input_t val_to_find{2};

    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(host_items.begin(), host_items.end(), expected_result[0], op_t{val_to_find});

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes{};

    // Run test
    c2h::device_vector<output_t> out_result(1);

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, in_items.begin(), out_result.begin(), op_t{val_to_find}, num_items);

    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, in_items.begin(), out_result.begin(), op_t{val_to_find}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("find_if works for unaligned input")
  {
    for (int offset = 1; offset < 4; ++offset)
    {
      if (num_items - offset > 0)
      {
        using op_t = equals<input_t>;
        input_t val_to_find{2};

        // Prepare verification data
        c2h::host_vector<input_t> host_items(in_items);
        c2h::host_vector<output_t> expected_result(1);
        compute_find_if_reference(host_items.begin() + offset, host_items.end(), expected_result[0], op_t{val_to_find});

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes{};

        // Run test
        c2h::device_vector<output_t> out_result(1);
        auto d_out_it = thrust::raw_pointer_cast(out_result.data());

        cub::DeviceFind::FindIf(
          d_temp_storage,
          temp_storage_bytes,
          unwrap_it(d_in_it + offset),
          unwrap_it(d_out_it),
          op_t{val_to_find},
          num_items - offset);

        thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
        d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

        cub::DeviceFind::FindIf(
          d_temp_storage,
          temp_storage_bytes,
          unwrap_it(d_in_it + offset),
          unwrap_it(d_out_it),
          op_t{val_to_find},
          num_items - offset);

        // Verify result
        REQUIRE(expected_result == out_result);
      }
    }
  }

  SECTION("find_if works with non primitive iterator")
  {
    using op_t = equals<input_t>;
    input_t val_to_find{2};

    // Prepare verification data
    auto it = thrust::make_counting_iterator(0); // non-primitive iterator
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(it, it + num_items, expected_result[0], op_t{val_to_find});

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes{};

    // Run test
    c2h::device_vector<output_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, it, unwrap_it(d_out_it), op_t{val_to_find}, num_items);

    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, it, unwrap_it(d_out_it), op_t{val_to_find}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }
}
