// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/iterator>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, find_if);

// List of types to test
using custom_t = c2h::custom_type_t<c2h::equal_comparable_t>;
using value_types =
  c2h::type_list<std::int8_t,
                 std::int16_t,
                 std::int32_t,
                 std::int64_t,
#if TEST_INT128()
                 __int128_t,
#endif // TEST_INT128()
                 custom_t>;
using offset_types = c2h::type_list<int32_t, int64_t>;

enum class gen_data_t
{
  GEN_TYPE_RANDOM, /// Uniform random data generation
  GEN_TYPE_CONST /// Constant value as input data
};

template <typename OffsetT, typename InputIt, typename Predicate>
auto compute_find_if_reference(InputIt first, InputIt last, Predicate predicate) -> OffsetT
{
  const auto it = std::find_if(first, last, predicate); // not thrust::find_if because it will rely on cub::FindIf
  return static_cast<OffsetT>(std::distance(first, it));
}

C2H_TEST("Device find_if works", "[device][find_if]", value_types, offset_types)
{
  using input_t  = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 10'000'000; // 10M items for reasonable test time
  // TODO(bgruber): test a value larger than UINT32

  // Generate the input sizes to test for
  const offset_t num_items =
    GENERATE_COPY(offset_t{1}, offset_t{100}, offset_t{5324}, max_items, take(5, random(min_items, max_items)));

  const gen_data_t data_gen_mode = GENERATE(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);
  const bool value_exists        = GENERATE(false, true);

  CAPTURE(c2h::type_name<input_t>(), c2h::type_name<offset_t>(), num_items, data_gen_mode, value_exists);

  constexpr bool is_custom_t = cuda::std::is_same_v<input_t, custom_t>;
  if constexpr (is_custom_t)
  {
    if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM && !value_exists)
    {
      //  min/max handling is not implemented for c2h::gen and custom_t, so we cannot pick a value that does not exist
      //  in the input sequence
      return;
    }
  }

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items, thrust::default_init);
  if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
  {
    if constexpr (is_custom_t)
    {
      c2h::gen(C2H_SEED(1), in_items);
    }
    else
    {
      // omit the largest value from the random values so we have a value to that does not occur
      c2h::gen(C2H_SEED(1), in_items, input_t{0}, static_cast<input_t>(::cuda::std::numeric_limits<input_t>::max() - 1));
    }
  }
  else
  {
    // fill with 1s
    input_t default_constant;
    init_default_constant(default_constant, 1);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  using predice_t = thrust::detail::equal_to_value<input_t>;
  input_t val_to_find{};

  // Generate test cases for both "found" and "not found" scenarios
  if (value_exists)
  {
    // take a random value from the input sequence
    val_to_find = in_items[GENERATE_COPY(take(1, random(offset_t{0}, num_items - 1)))];
  }
  else
  {
    // max value is neither in the random input and nor in the constant
    val_to_find = ::cuda::std::numeric_limits<input_t>::max();
  }

  auto predicate = predice_t{val_to_find};

  SECTION("Generic find if case")
  {
    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    const auto expected_result = compute_find_if_reference<offset_t>(host_items.begin(), host_items.end(), predicate);

    // Run test
    c2h::device_vector<offset_t> out_result(1, thrust::no_init);
    find_if(d_in_it, thrust::raw_pointer_cast(out_result.data()), predicate, num_items);
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("find_if works with thrust contiguous iterator")
  {
    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    const auto expected = compute_find_if_reference<offset_t>(host_items.begin(), host_items.end(), predicate);

    // Run test
    c2h::device_vector<offset_t> out_result(1);
    find_if(in_items.begin(), out_result.begin(), predicate, num_items);
    REQUIRE(expected == out_result[0]);
  }

  SECTION("find_if works for unaligned input")
  {
    for (int offset = 1; offset < 4; ++offset)
    {
      if (num_items > offset)
      {
        // Prepare verification data
        c2h::host_vector<input_t> host_items(in_items);
        const auto expected =
          compute_find_if_reference<offset_t>(host_items.begin() + offset, host_items.end(), predicate);

        // Run test
        c2h::device_vector<offset_t> out_result(1, thrust::no_init);
        find_if(d_in_it + offset, thrust::raw_pointer_cast(out_result.data()), predicate, num_items - offset);
        REQUIRE(expected == out_result[0]);
      }
    }
  }
}

C2H_TEST("Device find_if works with non primitive iterator", "[device][find_if]")
{
  using input_t  = int32_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 10000000; // 10M items for reasonable test time

  input_t val_to_find = static_cast<input_t>(GENERATE_COPY(take(1, random(min_items, max_items))));
  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(1, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  CAPTURE(num_items, val_to_find);

  const auto expected_if_found = cuda::std::min(static_cast<offset_t>(val_to_find), num_items);

  // counting_iterator input
  auto c_it = cuda::make_counting_iterator(input_t{0});
  {
    c2h::device_vector<offset_t> out_result(1, thrust::no_init);
    auto predicate = thrust::detail::equal_to_value<input_t>{val_to_find};
    find_if(c_it, thrust::raw_pointer_cast(out_result.data()), predicate, num_items);
    REQUIRE(expected_if_found == out_result[0]);
  }

  { // transform_iterator of counting_iterator input and thrust device_ptr output
    auto t_it = cuda::make_transform_iterator(c_it, ::cuda::std::negate{});
    c2h::device_vector<offset_t> out_result(1, thrust::no_init);
    auto predicate = thrust::detail::equal_to_value<input_t>{-val_to_find};
    find_if(t_it, out_result.data(), predicate, num_items);
    REQUIRE(expected_if_found == out_result[0]);
  }

  { // counting_iterator input and transform_output_iterator output
    c2h::device_vector<offset_t> out_result(1, thrust::no_init);
    auto predicate = thrust::detail::equal_to_value<input_t>{val_to_find};
    auto out_it    = cuda::make_transform_output_iterator(out_result.begin(), ::cuda::std::negate{});
    find_if(c_it, out_it, predicate, num_items);
    REQUIRE(-expected_if_found == out_result[0]);
  }
}
