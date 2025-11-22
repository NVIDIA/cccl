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
// %PARAM% TEST_TYPES types 0:1:2:3

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, find_if);

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t, std::int32_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t, std::int32_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3, std::int32_t>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a,
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4,
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   std::int32_t>>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t, std::int32_t>
#if TEST_HALF_T()
, type_pair<half_t, std::int32_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t, std::int32_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
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
  auto pos = thrust::find_if(first, last, op);
  result   = static_cast<OutputIt>(pos - first);
}

template <typename T>
struct equals
{
  T val;

  __device__ __host__ bool operator()(T i) const
  {
    return i == val;
  }

  // Accept any type convertible to T (for half_t/__half, bfloat16_t/__nv_bfloat16 compatibility)
  template <typename U, typename = typename ::cuda::std::enable_if<!::cuda::std::is_same<T, U>::value>::type>
  __device__ __host__ bool operator()(U i) const
  {
    return T(i) == val;
  }
};

C2H_TEST("Device find_if works", "[device]", full_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = output_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = std::numeric_limits<offset_t>::max() / 5; // make test run faster

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
    c2h::gen(C2H_SEED(2), in_items);
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
  if constexpr (::cuda::std::is_arithmetic_v<input_t>)
  {
    val_to_find = static_cast<input_t>(GENERATE_COPY(take(1, random(min_items, max_items))));
  }
  else
  {
    // For vector types, use a default constant value
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
  constexpr offset_t max_items = std::numeric_limits<offset_t>::max() / 5; // make test run faster

  using op_t          = equals<input_t>;
  input_t val_to_find = static_cast<input_t>(GENERATE_COPY(take(1, random(min_items, max_items))));
  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
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
