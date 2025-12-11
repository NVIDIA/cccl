// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_find_if.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include <nv/target>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1
// %PARAM% TEST_TYPES types 0:1

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, find_if);

// List of types to test
using custom_t = c2h::custom_type_t<c2h::equal_comparable_t>;

#if TEST_TYPES == 0
using value_types =
  c2h::type_list<std::uint8_t,
                 std::int8_t,
                 std::int16_t,
                 std::uint16_t,
                 std::int32_t,
                 std::uint32_t,
                 std::int64_t,
                 std::uint64_t,
                 float,
                 double>;
using offset_types = c2h::type_list<std::int32_t>;
#elif TEST_TYPES == 1
using value_types  = c2h::type_list<custom_t>;
using offset_types = c2h::type_list<std::int32_t>;
#endif

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
  auto pos = std::find_if(first, last, op); // not thrust::find_if because it will rely on cub::FindIf
  result   = static_cast<OutputIt>(std::distance(first, pos));
}

template <typename T>
struct equals
{
  T val;

  template <typename U>
  __device__ __host__ bool operator()(const U& i) const
  {
    return i == val;
  }
};

C2H_TEST("Device find_if works", "[device]", value_types, offset_types)
{
  using input_t  = c2h::get<0, TestType>;
  using output_t = c2h::get<1, TestType>;
  using offset_t = output_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 10000000; // 10M items for reasonable test time

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(1, random(min_items, max_items)),
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
    c2h::gen(C2H_SEED(1), in_items);
  }
  else
  {
    input_t default_constant{};
    init_default_constant(default_constant);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  using op_t = equals<input_t>;
  input_t val_to_find{};

  // Generate test cases for both "found" and "not found" scenarios
  enum class find_scenario_t : int
  {
    VALUE_EXISTS,
    VALUE_NOT_EXISTS
  };
  const find_scenario_t find_scenario = GENERATE_COPY(find_scenario_t::VALUE_EXISTS, find_scenario_t::VALUE_NOT_EXISTS);

  if constexpr (::cuda::std::is_arithmetic_v<input_t>)
  {
    if (find_scenario == find_scenario_t::VALUE_EXISTS)
    {
      // case where value exists in input
      val_to_find = in_items[GENERATE_COPY(take(1, random(0, num_items - 1)))];
    }
    else if (find_scenario == find_scenario_t::VALUE_NOT_EXISTS)
    {
      // case where value does not exist in input
      val_to_find = static_cast<input_t>(num_items + 1);
    }
  }
  else
  {
    // For vector types, use a default constant value
    // Note: find_scenario is not used here because we always use the default constant
    (void) find_scenario;
    init_default_constant(val_to_find);
  }

  SECTION("Generic find if case")
  {
    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(host_items.begin(), host_items.end(), expected_result[0], op_t{val_to_find});

    // Run test
    c2h::device_vector<output_t> out_result(1);
    output_t* d_out_it = thrust::raw_pointer_cast(out_result.data());

    find_if(unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{val_to_find}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("find_if works with non raw pointers - .begin() iterator")
  {
    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(host_items.begin(), host_items.end(), expected_result[0], op_t{val_to_find});

    // Run test
    c2h::device_vector<output_t> out_result(1);

    find_if(in_items.begin(), out_result.begin(), op_t{val_to_find}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("find_if works for unaligned input")
  {
    for (int offset = 1; offset < 4; ++offset)
    {
      if (num_items - offset > 0)
      {
        // Prepare verification data
        c2h::host_vector<input_t> host_items(in_items);
        c2h::host_vector<output_t> expected_result(1);
        compute_find_if_reference(host_items.begin() + offset, host_items.end(), expected_result[0], op_t{val_to_find});

        // Run test
        c2h::device_vector<output_t> out_result(1);
        auto d_out_it = thrust::raw_pointer_cast(out_result.data());

        find_if(unwrap_it(d_in_it + offset), unwrap_it(d_out_it), op_t{val_to_find}, num_items - offset);

        // Verify result
        REQUIRE(expected_result == out_result);
      }
    }
  }
}

C2H_TEST("Device find_if works with non primitive iterator",
         "[device]",
         c2h::type_list<type_pair<std::uint8_t, std::int32_t>>)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = output_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 10000000; // 10M items for reasonable test time

  using op_t          = equals<input_t>;
  input_t val_to_find = static_cast<input_t>(GENERATE_COPY(take(1, random(min_items, max_items))));
  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(1, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  SECTION("find_if works with non primitive iterator")
  {
    // Prepare verification data
    auto it = thrust::make_counting_iterator(0); // non-primitive iterator
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(it, it + num_items, expected_result[0], op_t{val_to_find});

    // Run test
    c2h::device_vector<output_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    find_if(it, unwrap_it(d_out_it), op_t{val_to_find}, num_items);
    // Verify result
    REQUIRE(expected_result == out_result);
  }
}
