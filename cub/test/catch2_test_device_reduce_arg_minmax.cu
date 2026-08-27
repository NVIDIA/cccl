// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <cuda/__cmath/uabs.h>
#include <cuda/devices>
#include <cuda/std/__algorithm/max_element.h>
#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinMax, device_arg_minmax);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinLastMax, device_arg_minlastmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

// clang-format off
using full_type_list = c2h::type_list<
  type_pair<std::uint8_t>,
  type_pair<std::int16_t, std::int32_t>, // DPX SIMD instructions and different (larger) output type
  type_pair<std::int32_t>,
  type_pair<std::int64_t, std::int32_t>, // different (smaller) output type
  type_pair<uchar3>,
  type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t>
#endif // TEST_HALF_T()
>;
// clang-format on

/**
 * @brief Input data generation mode
 */
enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

struct abs_less_t
{
  template <typename T>
  _CCCL_HOST_DEVICE_API auto operator()(const T& a, const T& b) const -> bool
  {
    return cuda::uabs(a) < cuda::uabs(b);
  }
};

template <typename... Args>
cudaError_t call_argminmax_api(bool last_max, Args&&... args)
{
  return last_max ? cub::DeviceReduce::ArgMinLastMax(::cuda::std::forward<Args>(args)...)
                  : cub::DeviceReduce::ArgMinMax(::cuda::std::forward<Args>(args)...);
}

template <typename... Args>
void call_argminmax_launch_wrapper(bool last_max, Args&&... args)
{
  if (last_max)
  {
    device_arg_minlastmax(::cuda::std::forward<Args>(args)...);
  }
  else
  {
    device_arg_minmax(::cuda::std::forward<Args>(args)...);
  }
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMin[Last]Max basic correctness", "[reduce][arg_minmax][arg_minlastmax]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = thrust::device_vector<int>{8, 6, -7, 5, 3, 1, -9};
  auto min_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(min_out[0] == -9);
  REQUIRE(min_index[0] == 6);
  REQUIRE(max_out[0] == 8);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMin[Last]Max handles zero-length input",
              "[reduce][arg_minmax][arg_minlastmax]",
              CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = thrust::device_vector<int>{};
  auto min_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  // For zero-length inputs, no output is written; just verify it does not crash.
  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMin[Last]Max handles single element",
              "[reduce][arg_minmax][arg_minlastmax]",
              CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = thrust::device_vector<int>{42};
  auto min_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(min_out[0] == 42);
  REQUIRE(min_index[0] == 0);
  REQUIRE(max_out[0] == 42);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMin[Last]Max with compare_op", "[reduce][arg_minmax][arg_minlastmax]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = thrust::device_vector<float>(1, thrust::no_init);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = thrust::device_vector<float>(1, thrust::no_init);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    cuda::std::less{});
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

// All-same values: first minimum at index 0. The maximum is the first occurrence (index 0) for ArgMinMax and the last
// occurrence (last index) for ArgMinLastMax.
CUB_TEST_CASE("cub::DeviceReduce::ArgMin[Last]Max tie-breaking", "[reduce][arg_minmax][arg_minlastmax]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = thrust::device_vector<int>{5, 5, 5, 5, 5};
  auto min_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = thrust::device_vector<int>(1, thrust::no_init);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(min_out[0] == 5);
  REQUIRE(min_index[0] == 0); // first minimum: smallest index on tie
  REQUIRE(max_out[0] == 5);
  REQUIRE(max_index[0] == (last_max ? 4 : 0));
}

CUB_TEST(
  "Device ArgMinMax and ArgMinLastMax works", "[reduce][device][arg_minmax][arg_minlastmax]", CUB_SMALL, full_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  // Generate input data
  c2h::device_vector<item_t> in_items(num_items);
  if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
  {
    c2h::gen(C2H_SEED(2), in_items);
  }
  else
  {
    item_t default_constant{};
    init_default_constant(default_constant);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  CAPTURE(c2h::type_name<item_t>(), c2h::type_name<output_t>(), num_items);

  constexpr int num_segments = 1;

  // Precompute reference values shared by both APIs
  c2h::host_vector<item_t> host_items(in_items);

  auto expected_min_it          = cuda::std::min_element(host_items.cbegin(), host_items.cend());
  const auto expected_min       = static_cast<output_t>(*expected_min_it);
  const auto expected_min_index = static_cast<cuda::std::int64_t>(expected_min_it - host_items.cbegin());

  // First maximum: standard max_element returns the first occurrence
  auto expected_first_max_it          = cuda::std::max_element(host_items.cbegin(), host_items.cend());
  const auto expected_first_max       = static_cast<output_t>(*expected_first_max_it);
  const auto expected_first_max_index = static_cast<cuda::std::int64_t>(expected_first_max_it - host_items.cbegin());

  // Last maximum: max_element on the reversed range finds the last occurrence in the forward range
  auto expected_last_max_it          = cuda::std::max_element(host_items.crbegin(), host_items.crend());
  const auto expected_last_max       = static_cast<output_t>(*expected_last_max_it);
  const auto expected_last_max_index = static_cast<cuda::std::int64_t>(host_items.size()) - 1
                                     - static_cast<cuda::std::int64_t>(expected_last_max_it - host_items.crbegin());

  using unwrapped_t = unwrap_value_t<output_t>;
  c2h::device_vector<unwrapped_t> d_min_out(num_segments), d_max_out(num_segments);
  c2h::device_vector<cuda::std::int64_t> d_min_index(num_segments), d_max_index(num_segments);

  SECTION("default")
  {
    const bool last_max           = GENERATE(false, true);
    const auto expected_max       = last_max ? expected_last_max : expected_first_max;
    const auto expected_max_index = last_max ? expected_last_max_index : expected_first_max_index;

    call_argminmax_launch_wrapper(
      last_max,
      unwrap_it(d_in_it),
      d_min_out.data(),
      d_min_index.data(),
      d_max_out.data(),
      d_max_index.data(),
      num_items);

    output_t gpu_min = static_cast<output_t>(d_min_out[0]);
    output_t gpu_max = static_cast<output_t>(d_max_out[0]);
    REQUIRE(expected_min == gpu_min);
    REQUIRE(expected_min_index == d_min_index[0]);
    REQUIRE(expected_max == gpu_max);
    REQUIRE(expected_max_index == d_max_index[0]);
  }

  // abs comparison via cuda::uabs only compiles for integral scalar types
  if constexpr (cuda::std::is_integral_v<item_t>)
  {
    SECTION("abs_less_t")
    {
      const bool last_max = GENERATE(false, true);
      abs_less_t compare_op;

      // First minimum by abs value: first element with smallest |value|
      auto exp_min_it          = cuda::std::min_element(host_items.cbegin(), host_items.cend(), compare_op);
      const auto exp_min       = static_cast<output_t>(*exp_min_it);
      const auto exp_min_index = static_cast<cuda::std::int64_t>(exp_min_it - host_items.cbegin());

      // First maximum by abs value: first element with largest |value|
      auto exp_first_max_it          = cuda::std::max_element(host_items.cbegin(), host_items.cend(), compare_op);
      const auto exp_first_max       = static_cast<output_t>(*exp_first_max_it);
      const auto exp_first_max_index = static_cast<cuda::std::int64_t>(exp_first_max_it - host_items.cbegin());

      // Last maximum by abs value: last element with largest |value|
      auto exp_last_max_it          = cuda::std::max_element(host_items.crbegin(), host_items.crend(), compare_op);
      const auto exp_last_max       = static_cast<output_t>(*exp_last_max_it);
      const auto exp_last_max_index = static_cast<cuda::std::int64_t>(host_items.size()) - 1
                                    - static_cast<cuda::std::int64_t>(exp_last_max_it - host_items.crbegin());

      const auto exp_max       = last_max ? exp_last_max : exp_first_max;
      const auto exp_max_index = last_max ? exp_last_max_index : exp_first_max_index;

      call_argminmax_launch_wrapper(
        last_max,
        unwrap_it(d_in_it),
        d_min_out.data(),
        d_min_index.data(),
        d_max_out.data(),
        d_max_index.data(),
        num_items,
        compare_op);

      output_t gpu_min = static_cast<output_t>(d_min_out[0]);
      output_t gpu_max = static_cast<output_t>(d_max_out[0]);
      REQUIRE(exp_min == gpu_min);
      REQUIRE(exp_min_index == d_min_index[0]);
      REQUIRE(exp_max == gpu_max);
      REQUIRE(exp_max_index == d_max_index[0]);
    }
  }
}
