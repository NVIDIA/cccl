// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
#include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

#include <cub/device/device_reduce.cuh>
#include <cub/util_type.cuh>

#include <thrust/sequence.h>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include "catch2_test_custom_streams.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/device_and_stream.h>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/test_util_vec.h>
#include <c2h/utility.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min_old);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max_old);
_CCCL_SUPPRESS_DEPRECATED_POP

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3:4

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t>
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t>
#endif // TEST_BF_T()
>;
// clang-format on
#elif TEST_TYPES == 4
// DPX SIMD instructions
using full_type_list = c2h::type_list<type_pair<std::uint16_t>, type_pair<std::int16_t>>;
#endif

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
    // need to use `uabs` to avoid integer overflow in case of abs(INT_MIN)
    return cuda::uabs(a) < cuda::uabs(b);
  }
};

CUB_TEST("Device reduce handles vectorized 16-bit input", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using input_t  = std::int16_t;
  using output_t = std::int32_t;

  constexpr int num_items = 8192;
  auto input              = c2h::make_device_buffer<input_t>(stream, num_items, input_t{1});
  auto output             = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);

  device_sum(input.data(), output.data(), num_items, stream);

  REQUIRE(c2h::read_single(stream, output) == num_items);
}

CUB_TEST("Device ArgMin and ArgMax compare input values before conversion", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using input_t  = float;
  using output_t = int;
  using index_t  = cuda::std::int64_t;

  SECTION("ArgMin")
  {
    auto input  = c2h::make_device_buffer<input_t>(stream, {1.9f, 1.2f});
    auto output = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
    auto index  = c2h::make_device_buffer<index_t>(stream, 1, cuda::no_init);

    device_arg_min(input.data(), output.data(), index.data(), static_cast<int>(input.size()), stream);

    REQUIRE(c2h::read_single(stream, output) == 1);
    REQUIRE(c2h::read_single(stream, index) == 1);
  }

  SECTION("ArgMax")
  {
    auto input  = c2h::make_device_buffer<input_t>(stream, {1.2f, 1.9f});
    auto output = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
    auto index  = c2h::make_device_buffer<index_t>(stream, 1, cuda::no_init);

    device_arg_max(input.data(), output.data(), index.data(), static_cast<int>(input.size()), stream);

    REQUIRE(c2h::read_single(stream, output) == 1);
    REQUIRE(c2h::read_single(stream, index) == 1);
  }
}

CUB_TEST("Device ArgMin and ArgMax handle empty input", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using input_t  = int;
  using index_t  = cuda::std::int64_t;
  using limits_t = cuda::std::numeric_limits<input_t>;

  constexpr input_t sentinel_value = 42;
  constexpr index_t sentinel_index = 42;

  auto input  = c2h::make_device_buffer<input_t>(stream, 1, input_t{7});
  auto output = c2h::make_device_buffer<input_t>(stream, 1, sentinel_value);
  auto index  = c2h::make_device_buffer<index_t>(stream, 1, sentinel_index);

  SECTION("ArgMin")
  {
    device_arg_min(input.data(), output.data(), index.data(), 0, stream);

    REQUIRE(c2h::read_single(stream, output) == limits_t::max());
    REQUIRE(c2h::read_single(stream, index) == 1);
  }

  SECTION("ArgMin with custom comparison operator")
  {
    device_arg_min(input.data(), output.data(), index.data(), 0, abs_less_t{}, stream);

    REQUIRE(c2h::read_single(stream, output) == input_t{});
    REQUIRE(c2h::read_single(stream, index) == 1);
  }

  SECTION("ArgMax")
  {
    device_arg_max(input.data(), output.data(), index.data(), 0, stream);

    REQUIRE(c2h::read_single(stream, output) == limits_t::lowest());
    REQUIRE(c2h::read_single(stream, index) == 1);
  }

  SECTION("ArgMax with custom comparison operator")
  {
    device_arg_max(input.data(), output.data(), index.data(), 0, abs_less_t{}, stream);

    REQUIRE(c2h::read_single(stream, output) == input_t{});
    REQUIRE(c2h::read_single(stream, index) == 1);
  }
}

CUB_TEST("Device reduce works with all device interfaces", "[reduce][device]", CUB_SMALL, full_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr int max_items    = 5000000;
  constexpr int min_items    = 1;
  constexpr int num_segments = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  const auto stream     = c2h::make_current_device_stream();
  const auto item_count = static_cast<std::size_t>(num_items);
  auto input_buffers    = [&] {
    if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
    {
      return c2h::gen_buffers<item_t>(stream, C2H_SEED(2), item_count);
    }

    item_t value{};
    init_default_constant(value);
    auto host_items   = c2h::make_host_buffer<item_t>(stream, item_count, value);
    auto device_items = c2h::make_device_buffer<item_t>(stream, host_items);
    return c2h::device_host_buffers<item_t>{cuda::std::move(device_items), cuda::std::move(host_items)};
  }();
  auto& in_items         = input_buffers.d_items;
  const auto& host_items = input_buffers.h_items;
  auto d_in_it           = in_items.data();

  CAPTURE(c2h::type_name<item_t>(), c2h::type_name<output_t>(), num_items);

#if TEST_TYPES != 4
  SECTION("reduce")
  {
    using op_t = cuda::std::plus<>;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    using accum_t                  = cuda::std::__accumulator_t<op_t, item_t, output_t>;
    const output_t expected_result = static_cast<output_t>(
      compute_single_problem_reference(host_items.cbegin(), host_items.cend(), reduction_op, accum_t{}));

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it      = out_result.data();
    using init_value_t = unwrap_value_t<output_t>;
    device_reduce(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items, reduction_op, init_value_t{}, stream);

    // Verify result
    REQUIRE(expected_result == c2h::read_single(stream, out_result));
  }
#endif // TEST_TYPES != 4

// Skip DeviceReduce::Sum tests for extended floating-point types because of unbounded epsilon due
// to pseudo associativity of the addition operation over floating point numbers
#if TEST_TYPES != 3
  SECTION("sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, item_t, output_t>;

    // Prepare verification data
    const output_t expected_result = static_cast<output_t>(
      compute_single_problem_reference(host_items.cbegin(), host_items.cend(), op_t{}, accum_t{}));

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = unwrap_it(out_result.data());
    device_sum(d_in_it, d_out_it, num_items, stream);

    // Verify result
    REQUIRE(expected_result == c2h::read_single(stream, out_result));
  }
#endif

  SECTION("min")
  {
    // Prepare verification data
    auto expected_result = *std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    device_min(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items, stream);

    // Verify result
    REQUIRE(expected_result == c2h::read_single(stream, out_result));
  }

  SECTION("max")
  {
    // Prepare verification data
    auto expected_result = *std::max_element(host_items.cbegin(), host_items.cend());

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    device_max(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items, stream);

    // Verify result
    REQUIRE(expected_result == c2h::read_single(stream, out_result));
  }

#if TEST_TYPES != 4
  SECTION("argmax")
  {
    // Prepare verification data
    auto expected_result = std::max_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;
    device_arg_max(unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, stream);

    // Verify result
    const result_t gpu_result   = c2h::read_single(stream, out_result);
    const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_extremum);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
  }

  SECTION("argmax with user provided memory and environment")
  {
    // Prepare verification data
    auto expected_result = std::max_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;

    size_t expected_allocation_size = 0;
    auto error                      = cub::DeviceReduce::ArgMax(
      static_cast<void*>(nullptr), expected_allocation_size, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp        = c2h::make_device_buffer<std::uint8_t>(stream, expected_allocation_size, cuda::no_init);
    void* temp_storage = d_temp.data();

    auto test_argmax = [&](const auto& env) {
      size_t num_bytes = 0;
      error            = cub::DeviceReduce::ArgMax(
        static_cast<void*>(nullptr), num_bytes, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
      REQUIRE(expected_allocation_size == num_bytes);

      error = cub::DeviceReduce::ArgMax(
        temp_storage, num_bytes, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());

      // Verify result
      const result_t gpu_result   = c2h::read_single(stream, out_result);
      const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
      REQUIRE(expected_result[0] == gpu_extremum);
      REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
    };

    stream.sync();
    test_with_custom_streams(test_argmax);
  }

  SECTION("argmin")
  {
    // Prepare verification data
    auto expected_result = std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;
    device_arg_min(unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, stream);

    // Verify result
    const result_t gpu_result   = c2h::read_single(stream, out_result);
    const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_extremum);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
  }

  SECTION("argmin with user provided memory and environment")
  {
    // Prepare verification data
    auto expected_result = std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;

    size_t expected_allocation_size = 0;
    auto error                      = cub::DeviceReduce::ArgMin(
      static_cast<void*>(nullptr), expected_allocation_size, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp        = c2h::make_device_buffer<std::uint8_t>(stream, expected_allocation_size, cuda::no_init);
    void* temp_storage = d_temp.data();

    auto test_argmin = [&](const auto& env) {
      size_t num_bytes = 0;
      error            = cub::DeviceReduce::ArgMin(
        static_cast<void*>(nullptr), num_bytes, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
      REQUIRE(expected_allocation_size == num_bytes);

      error = cub::DeviceReduce::ArgMin(
        temp_storage, num_bytes, unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());

      // Verify result
      const result_t gpu_result   = c2h::read_single(stream, out_result);
      const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
      REQUIRE(expected_result[0] == gpu_extremum);
      REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
    };

    stream.sync();
    test_with_custom_streams(test_argmin);
  }

  SECTION("argmax deprecated interface")
  {
    // Prepare verification data
    auto expected_result = std::max_element(host_items.cbegin(), host_items.cend());

    // Run test using the deprecated interface
    using result_t  = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    device_arg_max_old(unwrap_it(d_in_it), out_result.data(), num_items, stream);

    // Verify result for the deprecated interface
    const result_t gpu_result = c2h::read_single(stream, out_result);
    const output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_value);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  }

  SECTION("argmin deprecated interface")
  {
    // Prepare verification data
    auto expected_result = std::min_element(host_items.cbegin(), host_items.cend());

    // Run test using the deprecated interface
    using result_t  = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    device_arg_min_old(unwrap_it(d_in_it), out_result.data(), num_items, stream);

    // Verify result for the deprecated interface
    const result_t gpu_result = c2h::read_single(stream, out_result);
    const output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_value);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  }

#  if TEST_TYPES < 2
  SECTION("argmin-abs_less_t")
  {
    const abs_less_t compare_op;

    // Prepare verification data
    auto expected_result = cuda::std::min_element(host_items.cbegin(), host_items.cend(), compare_op);

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;
    device_arg_min(unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, compare_op, stream);

    // Verify result
    const result_t gpu_result   = c2h::read_single(stream, out_result);
    const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_extremum);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
  }

  SECTION("argmax-abs_less_t")
  {
    const abs_less_t compare_op;

    // Prepare verification data
    auto expected_result = cuda::std::max_element(host_items.cbegin(), host_items.cend(), compare_op);

    // Run test
    using result_t  = cuda::std::pair<cuda::std::int32_t, unwrap_value_t<output_t>>;
    auto out_result = c2h::make_device_buffer<result_t>(stream, num_segments, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_result_ptr   = out_result.data();
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;
    device_arg_max(unwrap_it(d_in_it), d_extremum_out, d_index_out, num_items, compare_op, stream);

    // Verify result
    const result_t gpu_result   = c2h::read_single(stream, out_result);
    const output_t gpu_extremum = static_cast<output_t>(gpu_result.second); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_extremum);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.first);
  }
#  endif
#endif
}

#if TEST_TYPES == 0
// this type stands in for lambda functions, which are also not copy-assignable before C++17
struct non_copy_assignable_plus
{
  non_copy_assignable_plus()                                           = default;
  non_copy_assignable_plus(const non_copy_assignable_plus&)            = default;
  non_copy_assignable_plus& operator=(const non_copy_assignable_plus&) = delete;

  template <typename T>
  _CCCL_HOST_DEVICE_API auto operator()(const T& a, const T& b) const -> T
  {
    return a + b;
  }
};

struct non_copy_assignable_less
{
  non_copy_assignable_less()                                           = default;
  non_copy_assignable_less(const non_copy_assignable_less&)            = default;
  non_copy_assignable_less& operator=(const non_copy_assignable_less&) = delete;

  template <typename T>
  _CCCL_HOST_DEVICE_API auto operator()(const T& a, const T& b) const -> bool
  {
    return a < b;
  }
};

CUB_TEST("Device reduce works with a non copy assignable reduction operator", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using item_t   = int;
  using output_t = int;

  constexpr int num_items = 1000;

  auto input = c2h::make_device_buffer<item_t>(stream, num_items, 42);
  thrust::sequence(c2h::device_policy.on(stream.get()), input.data(), input.data() + input.size(), 1);

  SECTION("reduce")
  {
    auto output = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
    device_reduce(input.data(), output.data(), num_items, non_copy_assignable_plus{}, 0, stream);
    CHECK((num_items * (num_items + 1)) / 2 == c2h::read_single(stream, output));
  }

  SECTION("argmin")
  {
    auto output_extremum = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
    auto output_index    = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);
    device_arg_min(
      input.data(), output_extremum.data(), output_index.data(), num_items, non_copy_assignable_less{}, stream);
    REQUIRE(1 == c2h::read_single(stream, output_extremum));
    REQUIRE(0 == c2h::read_single(stream, output_index));
  }
}

struct checking_reduce
{
  static constexpr auto sentinel = 42;

  _CCCL_HOST_DEVICE_API auto operator()(int a, int b) const -> int
  {
    CHECK(a == sentinel);
    CHECK(b == sentinel);
    return sentinel;
  }
};

struct faulting_reduce
{
  _CCCL_HOST_DEVICE_API auto operator()(int, int) const -> int
  {
    CHECK(false);
    return 0;
  }
};

CUB_TEST("Device reduce works without initial value", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr int num_items = 1000;
  auto input              = c2h::make_device_buffer<int>(stream, num_items, checking_reduce::sentinel);

  SECTION("for some elements")
  {
    auto output = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);
    device_reduce(input.data(), output.data(), num_items, checking_reduce{}, cub::detail::reduce::no_init, stream);
    CHECK(c2h::read_single(stream, output) == checking_reduce::sentinel);
  }
  SECTION("for no elements")
  {
    device_reduce(input.data(), static_cast<int*>(nullptr), 0, faulting_reduce{}, cub::detail::reduce::no_init, stream);
  }
}
#endif // TEST_TYPES == 0
