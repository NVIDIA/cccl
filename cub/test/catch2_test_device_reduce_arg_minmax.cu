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

CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax basic correctness", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{8, 6, -7, 5, 3, 1, -9};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  auto error = cub::DeviceReduce::ArgMinMax(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  error = cub::DeviceReduce::ArgMinMax(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == -9);
  REQUIRE(min_index[0] == 6);
  REQUIRE(max_out[0] == 8);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax basic correctness", "[reduce][arg_minlastmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{8, 6, -7, 5, 3, 1, -9};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  auto error = cub::DeviceReduce::ArgMinLastMax(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  error = cub::DeviceReduce::ArgMinLastMax(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == -9);
  REQUIRE(min_index[0] == 6);
  REQUIRE(max_out[0] == 8);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax handles zero-length input", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  // For zero-length inputs, no output is written; just verify it does not crash.
  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax handles zero-length input", "[reduce][arg_minlastmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  // For zero-length inputs, no output is written; just verify it does not crash.
  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax accepts stream", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = thrust::device_vector<float>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<float>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax accepts stream", "[reduce][arg_minlastmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = thrust::device_vector<float>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<float>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax handles single element", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{42};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 42);
  REQUIRE(min_index[0] == 0);
  REQUIRE(max_out[0] == 42);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax handles single element", "[reduce][arg_minlastmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{42};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 42);
  REQUIRE(min_index[0] == 0);
  REQUIRE(max_out[0] == 42);
  REQUIRE(max_index[0] == 0);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax with compare_op", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = thrust::device_vector<float>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<float>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    cuda::std::less{});

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax with compare_op", "[reduce][arg_minlastmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = thrust::device_vector<float>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<float>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    cuda::std::less{});

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

// All-same values: first minimum at index 0, first maximum at index 0
CUB_TEST_CASE("cub::DeviceReduce::ArgMinMax tie-breaking: first min and first max", "[reduce][arg_minmax]", CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{5, 5, 5, 5, 5};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 5);
  REQUIRE(min_index[0] == 0); // first minimum: smallest index on tie
  REQUIRE(max_out[0] == 5);
  REQUIRE(max_index[0] == 0); // first maximum: smallest index on tie
}

// All-same values: first minimum at index 0, last maximum at last index
CUB_TEST_CASE("cub::DeviceReduce::ArgMinLastMax tie-breaking: first min and last max",
              "[reduce][arg_minlastmax]",
              CUB_SMALL)
{
  auto input     = thrust::device_vector<int>{5, 5, 5, 5, 5};
  auto min_out   = thrust::device_vector<int>(1);
  auto min_index = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_out   = thrust::device_vector<int>(1);
  auto max_index = thrust::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 5);
  REQUIRE(min_index[0] == 0); // first minimum: smallest index on tie
  REQUIRE(max_out[0] == 5);
  REQUIRE(max_index[0] == 4); // last maximum: largest index on tie
}

CUB_TEST("Device ArgMinMax and ArgMinLastMax work with all device interfaces",
         "[reduce][device][arg_minmax][arg_minlastmax]",
         CUB_SMALL,
         full_type_list)
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

  // Precompute reference values shared by both sections
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

  SECTION("argminmax")
  {
    device_arg_minmax(
      unwrap_it(d_in_it), d_min_out.data(), d_min_index.data(), d_max_out.data(), d_max_index.data(), num_items);

    output_t gpu_min = static_cast<output_t>(d_min_out[0]);
    output_t gpu_max = static_cast<output_t>(d_max_out[0]);
    REQUIRE(expected_min == gpu_min);
    REQUIRE(expected_min_index == d_min_index[0]);
    REQUIRE(expected_first_max == gpu_max);
    REQUIRE(expected_first_max_index == d_max_index[0]);
  }

  SECTION("argminlastmax")
  {
    device_arg_minlastmax(
      unwrap_it(d_in_it), d_min_out.data(), d_min_index.data(), d_max_out.data(), d_max_index.data(), num_items);

    output_t gpu_min = static_cast<output_t>(d_min_out[0]);
    output_t gpu_max = static_cast<output_t>(d_max_out[0]);
    REQUIRE(expected_min == gpu_min);
    REQUIRE(expected_min_index == d_min_index[0]);
    REQUIRE(expected_last_max == gpu_max);
    REQUIRE(expected_last_max_index == d_max_index[0]);
  }

  // GCC7 ICEs (finish_member_declaration, cp/semantics.c:3029) on the generic lambda used below to
  // exercise the various environment/stream argument types, so skip this section there. The behavior
  // is compiler-independent and remains covered by newer compilers and the *_env test files.
#if !_CCCL_COMPILER(GCC, <, 8)
  SECTION("argminmax with user provided memory and environment")
  {
    const auto num_items_i64 = static_cast<cuda::std::int64_t>(num_items);

    size_t expected_allocation_size = 0;
    auto error                      = cub::DeviceReduce::ArgMinMax(
      static_cast<void*>(nullptr),
      expected_allocation_size,
      unwrap_it(d_in_it),
      d_min_out.data(),
      d_min_index.data(),
      d_max_out.data(),
      d_max_index.data(),
      num_items_i64);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
    void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

    auto test_argminmax = [&](const auto& env) {
      size_t num_bytes = 0;
      error            = cub::DeviceReduce::ArgMinMax(
        static_cast<void*>(nullptr),
        num_bytes,
        unwrap_it(d_in_it),
        d_min_out.data(),
        d_min_index.data(),
        d_max_out.data(),
        d_max_index.data(),
        num_items_i64,
        env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
      REQUIRE(expected_allocation_size == num_bytes);

      error = cub::DeviceReduce::ArgMinMax(
        temp_storage,
        num_bytes,
        unwrap_it(d_in_it),
        d_min_out.data(),
        d_min_index.data(),
        d_max_out.data(),
        d_max_index.data(),
        num_items_i64,
        env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());

      output_t gpu_min = static_cast<output_t>(d_min_out[0]);
      output_t gpu_max = static_cast<output_t>(d_max_out[0]);
      REQUIRE(expected_min == gpu_min);
      REQUIRE(expected_min_index == d_min_index[0]);
      REQUIRE(expected_first_max == gpu_max);
      REQUIRE(expected_first_max_index == d_max_index[0]);
    };

    int current_device;
    error = cudaGetDevice(&current_device);
    REQUIRE(error == cudaSuccess);

    SECTION("DeviceReduce::ArgMinMax works with cudaStream_t")
    {
      cuda::stream stream{cuda::devices[current_device]};
      test_argminmax(stream.get());
    }

    SECTION("DeviceReduce::ArgMinMax works with cuda::stream")
    {
      cuda::stream stream{cuda::devices[current_device]};
      test_argminmax(stream);
    }

    SECTION("DeviceReduce::ArgMinMax works with cuda::stream_ref")
    {
      cuda::stream stream{cuda::devices[current_device]};
      cuda::stream_ref stream_ref{stream};
      test_argminmax(stream_ref);
    }

    SECTION("DeviceReduce::ArgMinMax works with cuda::std::execution::env")
    {
      cuda::std::execution::env env{};
      test_argminmax(env);
    }

    SECTION("DeviceReduce::ArgMinMax works with cuda::execution::gpu")
    {
      const auto policy = cuda::execution::gpu;
      test_argminmax(policy);
    }

    SECTION("DeviceReduce::ArgMinMax works with cuda::execution::gpu with stream")
    {
      cuda::stream stream{cuda::devices[current_device]};
      const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
      test_argminmax(policy);
    }
  }

  SECTION("argminlastmax with user provided memory and environment")
  {
    const auto num_items_i64 = static_cast<cuda::std::int64_t>(num_items);

    size_t expected_allocation_size = 0;
    auto error                      = cub::DeviceReduce::ArgMinLastMax(
      static_cast<void*>(nullptr),
      expected_allocation_size,
      unwrap_it(d_in_it),
      d_min_out.data(),
      d_min_index.data(),
      d_max_out.data(),
      d_max_index.data(),
      num_items_i64);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
    void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

    auto test_argminlastmax = [&](const auto& env) {
      size_t num_bytes = 0;
      error            = cub::DeviceReduce::ArgMinLastMax(
        static_cast<void*>(nullptr),
        num_bytes,
        unwrap_it(d_in_it),
        d_min_out.data(),
        d_min_index.data(),
        d_max_out.data(),
        d_max_index.data(),
        num_items_i64,
        env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
      REQUIRE(expected_allocation_size == num_bytes);

      error = cub::DeviceReduce::ArgMinLastMax(
        temp_storage,
        num_bytes,
        unwrap_it(d_in_it),
        d_min_out.data(),
        d_min_index.data(),
        d_max_out.data(),
        d_max_index.data(),
        num_items_i64,
        env);
      REQUIRE(error == cudaSuccess);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());

      output_t gpu_min = static_cast<output_t>(d_min_out[0]);
      output_t gpu_max = static_cast<output_t>(d_max_out[0]);
      REQUIRE(expected_min == gpu_min);
      REQUIRE(expected_min_index == d_min_index[0]);
      REQUIRE(expected_last_max == gpu_max);
      REQUIRE(expected_last_max_index == d_max_index[0]);
    };

    int current_device;
    error = cudaGetDevice(&current_device);
    REQUIRE(error == cudaSuccess);

    SECTION("DeviceReduce::ArgMinLastMax works with cudaStream_t")
    {
      cuda::stream stream{cuda::devices[current_device]};
      test_argminlastmax(stream.get());
    }

    SECTION("DeviceReduce::ArgMinLastMax works with cuda::stream")
    {
      cuda::stream stream{cuda::devices[current_device]};
      test_argminlastmax(stream);
    }

    SECTION("DeviceReduce::ArgMinLastMax works with cuda::stream_ref")
    {
      cuda::stream stream{cuda::devices[current_device]};
      cuda::stream_ref stream_ref{stream};
      test_argminlastmax(stream_ref);
    }

    SECTION("DeviceReduce::ArgMinLastMax works with cuda::std::execution::env")
    {
      cuda::std::execution::env env{};
      test_argminlastmax(env);
    }

    SECTION("DeviceReduce::ArgMinLastMax works with cuda::execution::gpu")
    {
      const auto policy = cuda::execution::gpu;
      test_argminlastmax(policy);
    }

    SECTION("DeviceReduce::ArgMinLastMax works with cuda::execution::gpu with stream")
    {
      cuda::stream stream{cuda::devices[current_device]};
      const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
      test_argminlastmax(policy);
    }
  }
#endif // !_CCCL_COMPILER(GCC, <, 8)

  // abs comparison via cuda::uabs only compiles for integral scalar types
  if constexpr (cuda::std::is_integral_v<item_t>)
  {
    SECTION("argminmax-abs_less_t")
    {
      abs_less_t compare_op;

      // Prepare verification data
      c2h::host_vector<item_t> host_items_abs(in_items);

      // First minimum by abs value: first element with smallest |value|
      auto exp_min_it          = cuda::std::min_element(host_items_abs.cbegin(), host_items_abs.cend(), compare_op);
      const auto exp_min       = static_cast<output_t>(*exp_min_it);
      const auto exp_min_index = static_cast<cuda::std::int64_t>(exp_min_it - host_items_abs.cbegin());

      // First maximum by abs value: first element with largest |value|
      auto exp_first_max_it    = cuda::std::max_element(host_items_abs.cbegin(), host_items_abs.cend(), compare_op);
      const auto exp_first_max = static_cast<output_t>(*exp_first_max_it);
      const auto exp_first_max_index = static_cast<cuda::std::int64_t>(exp_first_max_it - host_items_abs.cbegin());

      device_arg_minmax(
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
      REQUIRE(exp_first_max == gpu_max);
      REQUIRE(exp_first_max_index == d_max_index[0]);
    }

    SECTION("argminlastmax-abs_less_t")
    {
      abs_less_t compare_op;

      // Prepare verification data
      c2h::host_vector<item_t> host_items_abs(in_items);

      // First minimum by abs value: first element with smallest |value|
      auto exp_min_it          = cuda::std::min_element(host_items_abs.cbegin(), host_items_abs.cend(), compare_op);
      const auto exp_min       = static_cast<output_t>(*exp_min_it);
      const auto exp_min_index = static_cast<cuda::std::int64_t>(exp_min_it - host_items_abs.cbegin());

      // Last maximum by abs value: last element with largest |value|
      auto exp_last_max_it    = cuda::std::max_element(host_items_abs.crbegin(), host_items_abs.crend(), compare_op);
      const auto exp_last_max = static_cast<output_t>(*exp_last_max_it);
      const auto exp_last_max_index = static_cast<cuda::std::int64_t>(host_items_abs.size()) - 1
                                    - static_cast<cuda::std::int64_t>(exp_last_max_it - host_items_abs.crbegin());

      device_arg_minlastmax(
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
      REQUIRE(exp_last_max == gpu_max);
      REQUIRE(exp_last_max_index == d_max_index[0]);
    }
  }
}
