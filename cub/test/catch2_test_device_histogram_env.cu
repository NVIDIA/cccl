// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_histogram.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/array>
#include <cuda/std/execution>

#include <sstream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceHistogram::HistogramEven, histogram_even);
DECLARE_LAUNCH_WRAPPER(cub::DeviceHistogram::HistogramRange, histogram_range);

DECLARE_TMPL_LAUNCH_WRAPPER(cub::DeviceHistogram::MultiHistogramEven,
                            multi_histogram_even,
                            ESCAPE_LIST(int Channels, int ActiveChannels),
                            ESCAPE_LIST(Channels, ActiveChannels));

DECLARE_TMPL_LAUNCH_WRAPPER(cub::DeviceHistogram::MultiHistogramRange,
                            multi_histogram_range,
                            ESCAPE_LIST(int Channels, int ActiveChannels),
                            ESCAPE_LIST(Channels, ActiveChannels));

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceHistogram::HistogramEven works with default environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<int>{0, 2, 1, 0, 3, 4, 2, 1};
  int num_samples  = static_cast<int>(d_samples.size());
  int num_levels   = 6;
  int lower_level  = 0;
  int upper_level  = 5;
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_samples));

  c2h::device_vector<int> expected{2, 2, 2, 1, 1};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramEven works with user provided memory and environment",
              "[histogram][device]",
              CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<int>{0, 2, 1, 0, 3, 4, 2, 1};
  int num_samples  = static_cast<int>(d_samples.size());
  int num_levels   = 6;
  int lower_level  = 0;
  int upper_level  = 5;
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  c2h::device_vector<int> expected{2, 2, 2, 1, 1};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceHistogram::HistogramEven(
    nullptr,
    expected_bytes_allocated,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_samples);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_histogram_even = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceHistogram::HistogramEven(
      nullptr,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_samples,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_bytes_allocated == num_bytes);

    error = cub::DeviceHistogram::HistogramEven(
      temp_storage,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_samples,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Verify result
    REQUIRE(d_histogram == expected);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceHistogram::HistogramEven works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_histogram_even(stream.get());
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_histogram_even(stream);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_histogram_even(stream_ref);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_histogram_even(env);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_histogram_even(policy);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_histogram_even(policy);
  }
}

CUB_TEST_CASE("DeviceHistogram::HistogramRange works with default environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<float>{2.2f, 6.1f, 7.5f, 2.9f, 3.5f, 0.3f, 2.9f, 2.1f};
  int num_samples  = static_cast<int>(d_samples.size());
  auto d_levels    = c2h::device_vector<float>{0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  REQUIRE(cudaSuccess
          == cub::DeviceHistogram::HistogramRange(
            thrust::raw_pointer_cast(d_samples.data()),
            thrust::raw_pointer_cast(d_histogram.data()),
            num_levels,
            thrust::raw_pointer_cast(d_levels.data()),
            num_samples));

  c2h::device_vector<int> expected{1, 5, 0, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramRange works with user provided memory and environment",
              "[histogram][device]",
              CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<float>{2.2f, 6.1f, 7.5f, 2.9f, 3.5f, 0.3f, 2.9f, 2.1f};
  int num_samples  = static_cast<int>(d_samples.size());
  auto d_levels    = c2h::device_vector<float>{0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  c2h::device_vector<int> expected{1, 5, 0, 2};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceHistogram::HistogramRange(
    nullptr,
    expected_bytes_allocated,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_samples);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_histogram_range = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceHistogram::HistogramRange(
      nullptr,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_samples,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_bytes_allocated == num_bytes);

    error = cub::DeviceHistogram::HistogramRange(
      temp_storage,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_samples,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Verify result
    REQUIRE(d_histogram == expected);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceHistogram::HistogramRange works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_histogram_range(stream.get());
  }

  SECTION("DeviceHistogram::HistogramRange works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_histogram_range(stream);
  }

  SECTION("DeviceHistogram::HistogramRange works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_histogram_range(stream_ref);
  }

  SECTION("DeviceHistogram::HistogramRange works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_histogram_range(env);
  }

  SECTION("DeviceHistogram::HistogramRange works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_histogram_range(policy);
  }

  SECTION("DeviceHistogram::HistogramRange works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_histogram_range(policy);
  }
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramEven works with default environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 pixels: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128)
  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  REQUIRE(cudaSuccess
          == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, lower_level, upper_level, num_pixels));

  c2h::device_vector<int> expected_r{1, 0, 0, 1};
  c2h::device_vector<int> expected_g{0, 0, 1, 0};
  c2h::device_vector<int> expected_b{0, 1, 1, 0};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramRange works with default environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 pixels: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128)
  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  REQUIRE(cudaSuccess
          == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, d_levels, num_pixels));

  c2h::device_vector<int> expected_r{1, 1};
  c2h::device_vector<int> expected_g{1, 1};
  c2h::device_vector<int> expected_b{0, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::HistogramEven 2D works with default environment", "[histogram][device]", CUB_SMALL)
{
  // 2 rows, 3 samples per row, stride of 4 (1 padding element)
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  int num_levels          = 4;
  int lower_level         = 0;
  int upper_level         = 3;
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_row_samples,
      num_rows,
      row_stride_bytes));

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramRange 2D works with default environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  auto d_levels           = c2h::device_vector<int>{0, 1, 2, 3};
  int num_levels          = static_cast<int>(d_levels.size());
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramRange(
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_row_samples,
      num_rows,
      row_stride_bytes));

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramEven 2D works with default environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 rows, 2 pixels per row, stride includes 1 extra pixel of padding
  // Row 0: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128), (PAD, PAD, PAD, PAD)
  // Row 1: (R=1, G=1, B=3, A=200), (R=2, G=3, B=0, A=100), (PAD, PAD, PAD, PAD)
  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char); // 3 pixels wide, 2 used

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  // R: 0,3,1,2 → bin[0]=1, bin[1]=1, bin[2]=1, bin[3]=1
  c2h::device_vector<int> expected_r{1, 1, 1, 1};
  // G: 2,4,1,3 → bin[1]=1, bin[2]=1, bin[3]=1 (4 out of range)
  c2h::device_vector<int> expected_g{0, 1, 1, 1};
  // B: 1,2,3,0 → bin[0]=1, bin[1]=1, bin[2]=1, bin[3]=1
  c2h::device_vector<int> expected_b{1, 1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramRange 2D works with default environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // Same layout as MultiHistogramEven 2D test
  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  // R: 0,3,1,2 → [0,2)=2, [2,4)=2
  c2h::device_vector<int> expected_r{2, 2};
  // G: 2,4,1,3 → [0,3)=2, [3,5)=2
  c2h::device_vector<int> expected_g{2, 2};
  // B: 1,2,3,0 → [0,1)=1, [1,2)=1, [2,3)=1 (3 out of range)
  c2h::device_vector<int> expected_b{1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

#endif

CUB_TEST("DeviceHistogram::HistogramEven uses environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<int>{0, 2, 1, 0, 3, 4, 2, 1};
  int num_samples  = static_cast<int>(d_samples.size());
  int num_levels   = 6;
  int lower_level  = 0;
  int upper_level  = 5;
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_samples));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  histogram_even(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_samples,
    env);

  c2h::device_vector<int> expected{2, 2, 2, 1, 1};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramEven uses custom stream", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<int>{0, 2, 1, 0, 3, 4, 2, 1};
  int num_samples  = static_cast<int>(d_samples.size());
  int num_levels   = 6;
  int lower_level  = 0;
  int upper_level  = 5;
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_samples));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  histogram_even(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_samples,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected{2, 2, 2, 1, 1};
  REQUIRE(d_histogram == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceHistogram::HistogramRange uses environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<float>{2.2f, 6.1f, 7.5f, 2.9f, 3.5f, 0.3f, 2.9f, 2.1f};
  int num_samples  = static_cast<int>(d_samples.size());
  auto d_levels    = c2h::device_vector<float>{0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramRange(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_samples));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  histogram_range(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_samples,
    env);

  c2h::device_vector<int> expected{1, 5, 0, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramRange uses custom stream", "[histogram][device]", CUB_SMALL)
{
  auto d_samples   = c2h::device_vector<float>{2.2f, 6.1f, 7.5f, 2.9f, 3.5f, 0.3f, 2.9f, 2.1f};
  int num_samples  = static_cast<int>(d_samples.size());
  auto d_levels    = c2h::device_vector<float>{0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramRange(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_samples));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  histogram_range(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_samples,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected{1, 5, 0, 2};
  REQUIRE(d_histogram == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceHistogram::MultiHistogramEven uses environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_pixels));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, lower_level, upper_level, num_pixels, env);

  c2h::device_vector<int> expected_r{1, 0, 0, 1};
  c2h::device_vector<int> expected_g{0, 0, 1, 0};
  c2h::device_vector<int> expected_b{0, 1, 1, 0};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramEven uses custom stream", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_pixels));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, lower_level, upper_level, num_pixels, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_r{1, 0, 0, 1};
  c2h::device_vector<int> expected_g{0, 0, 1, 0};
  c2h::device_vector<int> expected_b{0, 1, 1, 0};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

#if TEST_LAUNCH == 0
CUB_TEST("DeviceHistogram::MultiHistogramRange works with user provided memory and environment",
         "[histogram][device]",
         CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  c2h::device_vector<int> expected_r{1, 1};
  c2h::device_vector<int> expected_g{1, 1};
  c2h::device_vector<int> expected_b{0, 1, 1};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    nullptr,
    expected_bytes_allocated,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_pixels);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_multi_histogram_range = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr, num_bytes, thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, d_levels, num_pixels, env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_bytes_allocated == num_bytes);

    error = cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      temp_storage,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_pixels,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Verify result
    REQUIRE(d_histogram_r == expected_r);
    REQUIRE(d_histogram_g == expected_g);
    REQUIRE(d_histogram_b == expected_b);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceHistogram::MultiHistogramRange works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_multi_histogram_range(stream.get());
  }

  SECTION("DeviceHistogram::MultiHistogramRange works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_multi_histogram_range(stream);
  }

  SECTION("DeviceHistogram::MultiHistogramRange works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_multi_histogram_range(stream_ref);
  }

  SECTION("DeviceHistogram::MultiHistogramRange works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_multi_histogram_range(env);
  }

  SECTION("DeviceHistogram::MultiHistogramRange works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_multi_histogram_range(policy);
  }

  SECTION("DeviceHistogram::MultiHistogramRange works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_multi_histogram_range(policy);
  }
}
#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceHistogram::MultiHistogramRange uses environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_pixels));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, d_levels, num_pixels, env);

  c2h::device_vector<int> expected_r{1, 1};
  c2h::device_vector<int> expected_g{1, 1};
  c2h::device_vector<int> expected_b{0, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramRange uses custom stream", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples = c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_pixels));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, d_levels, num_pixels, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_r{1, 1};
  c2h::device_vector<int> expected_g{1, 1};
  c2h::device_vector<int> expected_b{0, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceHistogram::HistogramEven 2D uses environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  int num_levels          = 4;
  int lower_level         = 0;
  int upper_level         = 3;
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_row_samples,
      num_rows,
      row_stride_bytes));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  histogram_even(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramEven 2D uses custom stream", "[histogram][device]", CUB_SMALL)
{
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  int num_levels          = 4;
  int lower_level         = 0;
  int upper_level         = 3;
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramEven(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      lower_level,
      upper_level,
      num_row_samples,
      num_rows,
      row_stride_bytes));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  histogram_even(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceHistogram::HistogramRange 2D uses environment", "[histogram][device]", CUB_SMALL)
{
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  auto d_levels           = c2h::device_vector<int>{0, 1, 2, 3};
  int num_levels          = static_cast<int>(d_levels.size());
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramRange(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_row_samples,
      num_rows,
      row_stride_bytes));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  histogram_range(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);
}

CUB_TEST_CASE("DeviceHistogram::HistogramRange 2D uses custom stream", "[histogram][device]", CUB_SMALL)
{
  auto d_samples          = c2h::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  auto d_levels           = c2h::device_vector<int>{0, 1, 2, 3};
  int num_levels          = static_cast<int>(d_levels.size());
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);
  auto d_histogram        = c2h::device_vector<int>(num_levels - 1, 0);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::HistogramRange(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      thrust::raw_pointer_cast(d_histogram.data()),
      num_levels,
      thrust::raw_pointer_cast(d_levels.data()),
      num_row_samples,
      num_rows,
      row_stride_bytes));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  histogram_range(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected{2, 2, 2};
  REQUIRE(d_histogram == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceHistogram::MultiHistogramEven 2D uses environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);

  c2h::device_vector<int> expected_r{1, 1, 1, 1};
  c2h::device_vector<int> expected_g{0, 1, 1, 1};
  c2h::device_vector<int> expected_b{1, 1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramEven 2D uses custom stream", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_r{1, 1, 1, 1};
  c2h::device_vector<int> expected_g{0, 1, 1, 1};
  c2h::device_vector<int> expected_b{1, 1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

#if TEST_LAUNCH == 0
CUB_TEST_CASE("DeviceHistogram::MultiHistogramEven works with user provided memory and environment",
              "[histogram][device]",
              CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  c2h::device_vector<int> expected_r{1, 1, 1, 1};
  c2h::device_vector<int> expected_g{0, 1, 1, 1};
  c2h::device_vector<int> expected_b{1, 1, 1, 1};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    nullptr,
    expected_bytes_allocated,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_multi_histogram_even = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_bytes_allocated == num_bytes);

    error = cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      temp_storage,
      num_bytes,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Verify result
    REQUIRE(d_histogram_r == expected_r);
    REQUIRE(d_histogram_g == expected_g);
    REQUIRE(d_histogram_b == expected_b);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceHistogram::HistogramEven works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_multi_histogram_even(stream.get());
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_multi_histogram_even(stream);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_multi_histogram_even(stream_ref);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_multi_histogram_even(env);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_multi_histogram_even(policy);
  }

  SECTION("DeviceHistogram::HistogramEven works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_multi_histogram_even(policy);
  }
}
#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceHistogram::MultiHistogramRange 2D uses environment", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);

  c2h::device_vector<int> expected_r{2, 2};
  c2h::device_vector<int> expected_g{2, 2};
  c2h::device_vector<int> expected_b{1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

CUB_TEST_CASE("DeviceHistogram::MultiHistogramRange 2D uses custom stream", "[histogram][device]", CUB_SMALL)
{
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  auto d_samples =
    c2h::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char);

  auto d_levels_r = c2h::device_vector<unsigned char>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<unsigned char>{0, 3, 5};
  auto d_levels_b = c2h::device_vector<unsigned char>{0, 1, 2, 3};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(d_samples.data()),
      d_histogram,
      num_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_bytes));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_r{2, 2};
  c2h::device_vector<int> expected_g{2, 2};
  c2h::device_vector<int> expected_b{1, 1, 1};
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

#if TEST_LAUNCH != 1

template <int BlockThreads>
struct histogram_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::HistogramPolicy
  {
    return {
      BlockThreads,
      1,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      BlockThreads,
      1,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      256 * sizeof(unsigned int),
      0,
      BlockThreads,
      1,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      0,
      0,
      0,
      0,
      0,
      0};
  }
};

template <int BlockThreads, typename LocalCounterT>
struct histogram_tuning_with_local_counter : histogram_tuning<BlockThreads>
{
  using local_counter_type = LocalCounterT;
};

struct mixed_counter_histogram_tuning
{
  using local_counter_type = unsigned int;

  _CCCL_API constexpr auto operator()(cuda::compute_capability) const -> cub::HistogramPolicy
  {
    return {
      128,
      4,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      128,
      4,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      512 * sizeof(unsigned int),
      0,
      128,
      4,
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      false,
      false,
      228352,
      2048,
      28544,
      19029,
      8192,
      0};
  }
};

static_assert(
  cuda::std::is_same_v<
    cub::detail::histogram::local_counter_t<histogram_tuning_with_local_counter<128, unsigned int>, unsigned long long>,
    unsigned int>);
static_assert(cuda::std::is_same_v<cub::detail::histogram::local_counter_t<histogram_tuning<128>, unsigned long long>,
                                   unsigned long long>);

C2H_TEST("DeviceHistogram supports narrower local counters than output counters", "[histogram][device]")
{
  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc, current_device));
  if (cc < cuda::compute_capability{10, 0})
  {
    SKIP("The runtime-sized shared-memory histogram policy is currently tuned for SM100");
  }

  constexpr int num_samples = 4096;
  constexpr int num_levels  = num_samples + 1;
  auto d_histogram          = c2h::device_vector<unsigned long long>(num_samples, 0);
  auto env                  = cuda::execution::tune(mixed_counter_histogram_tuning{});

  histogram_even(
    cuda::counting_iterator<unsigned int>(0),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    0u,
    static_cast<unsigned int>(num_samples),
    num_samples,
    env);

  REQUIRE(d_histogram == c2h::host_vector<unsigned long long>(num_samples, 1));
}

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

CUB_TEST("DeviceHistogram::HistogramEven can be tuned", "[histogram][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_constant_iterator d_samples(0, thrust::raw_pointer_cast(d_block_size.data()));
  int num_samples  = 256;
  int num_levels   = 257;
  int lower_level  = 0;
  int upper_level  = 256;
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  auto env = cuda::execution::tune(histogram_tuning<target_block_size>{});

  histogram_even(
    d_samples, thrust::raw_pointer_cast(d_histogram.data()), num_levels, lower_level, upper_level, num_samples, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceHistogram::HistogramRange can be tuned", "[histogram][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_constant_iterator d_samples(0, thrust::raw_pointer_cast(d_block_size.data()));
  int num_samples  = 256;
  auto d_levels    = c2h::device_vector<int>{0, 128, 256};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = c2h::device_vector<int>(num_levels - 1, 0);

  auto env = cuda::execution::tune(histogram_tuning<target_block_size>{});

  histogram_range(
    d_samples,
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_samples,
    env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceHistogram::MultiHistogramEven can be tuned", "[histogram][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  constexpr int NUM_CHANNELS               = 4;
  constexpr int NUM_ACTIVE_CHANNELS        = 3;

  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_constant_iterator d_samples(0, thrust::raw_pointer_cast(d_block_size.data()));
  int num_pixels = 64;

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels  = {5, 5, 5};
  cuda::std::array<int, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<int, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = c2h::device_vector<int>(4, 0);
  auto d_histogram_g = c2h::device_vector<int>(4, 0);
  auto d_histogram_b = c2h::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  auto env = cuda::execution::tune(histogram_tuning<target_block_size>{});

  multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceHistogram::MultiHistogramRange can be tuned", "[histogram][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  constexpr int NUM_CHANNELS               = 4;
  constexpr int NUM_ACTIVE_CHANNELS        = 3;

  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_constant_iterator d_samples(0, thrust::raw_pointer_cast(d_block_size.data()));
  int num_pixels = 64;

  auto d_levels_r = c2h::device_vector<int>{0, 2, 4};
  auto d_levels_g = c2h::device_vector<int>{0, 2, 4};
  auto d_levels_b = c2h::device_vector<int>{0, 2, 4};

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 3};

  cuda::std::array<const int*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = c2h::device_vector<int>(2, 0);
  auto d_histogram_g = c2h::device_vector<int>(2, 0);
  auto d_histogram_b = c2h::device_vector<int>(2, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  auto env = cuda::execution::tune(histogram_tuning<target_block_size>{});

  multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    d_samples, d_histogram, num_levels, d_levels, num_pixels, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test HistogramPolicy properties", "[histogram][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::HistogramPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::HistogramPolicy>);

  // aggregate init
  constexpr auto p1 = cub::HistogramPolicy{
    128,
    7,
    4,
    cub::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_LDG,
    false,
    false,
    96,
    3,
    4,
    cub::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_LDG,
    false,
    false,
    2052,
    2,
    128,
    7,
    4,
    cub::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_LDG,
    false,
    false,
    12345,
    1024,
    4096,
    8192,
    16384,
    2048};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::HistogramPolicy{
    .gmem_threads_per_block            = 128,
    .gmem_items_per_thread             = 7,
    .gmem_vec_size                     = 4,
    .gmem_load_algorithm               = cub::BLOCK_LOAD_DIRECT,
    .gmem_load_modifier                = cub::CacheLoadModifier::LOAD_LDG,
    .gmem_rle_compress                 = false,
    .gmem_work_stealing                = false,
    .static_smem_threads_per_block     = 96,
    .static_smem_items_per_thread      = 3,
    .static_smem_vec_size              = 4,
    .static_smem_load_algorithm        = cub::BLOCK_LOAD_DIRECT,
    .static_smem_load_modifier         = cub::CacheLoadModifier::LOAD_LDG,
    .static_smem_rle_compress          = false,
    .static_smem_work_stealing         = false,
    .static_smem_max_privatized_bytes  = 2052,
    .static_smem_min_blocks_per_sm     = 2,
    .dynamic_smem_threads_per_block    = 128,
    .dynamic_smem_items_per_thread     = 7,
    .dynamic_smem_vec_size             = 4,
    .dynamic_smem_load_algorithm       = cub::BLOCK_LOAD_DIRECT,
    .dynamic_smem_load_modifier        = cub::CacheLoadModifier::LOAD_LDG,
    .dynamic_smem_rle_compress         = false,
    .dynamic_smem_work_stealing        = false,
    .dynamic_smem_max_privatized_bytes = 12345,
    .dynamic_smem_range_max_bins       = 1024,
    .dynamic_smem_even_2ch_max_bins    = 4096,
    .dynamic_smem_even_3ch_max_bins    = 8192,
    .dynamic_smem_even_4ch_max_bins    = 16384,
    .init_kernel_pdl_trigger_max_bins  = 2048};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  std::ostringstream os;
  os << p1;
  REQUIRE(os.str().find("HistogramPolicy { .gmem_threads_per_block = 128") == 0);
  REQUIRE(os.str().find(".static_smem_max_privatized_bytes = 2052") != std::string::npos);
  REQUIRE(os.str().find(".dynamic_smem_max_privatized_bytes = 12345") != std::string::npos);
}

C2H_TEST("Histogram SM100 policy carries the tuned dynamic shared-memory budget", "[histogram][device]")
{
  using selector_t = cub::detail::histogram::policy_selector_from_types<int, unsigned int, 1, 1, true>;

  constexpr auto sm90_policy  = selector_t{}(cuda::compute_capability{9, 0});
  constexpr auto sm100_policy = selector_t{}(cuda::compute_capability{10, 0});
  constexpr auto sm100_wide_counter_policy =
    cub::detail::histogram::policy_selector_from_types<int, unsigned long long, 1, 1, true>{}(
      cuda::compute_capability{10, 0});

  STATIC_REQUIRE(cub::detail::histogram::max_privatized_static_smem_bins(sm90_policy, 4, 1) == 256);
  STATIC_REQUIRE(cub::detail::histogram::max_privatized_static_smem_bins(sm100_policy, 4, 1) == 512);
  STATIC_REQUIRE(cub::detail::histogram::max_privatized_static_smem_bins(sm100_policy, 4, 4) == 128);
  STATIC_REQUIRE(sm90_policy.dynamic_smem_max_privatized_bytes == 0);
  STATIC_REQUIRE(sm100_policy.dynamic_smem_max_privatized_bytes == 228352);
  STATIC_REQUIRE(sm100_wide_counter_policy.dynamic_smem_max_privatized_bytes == 0);
  STATIC_REQUIRE(cub::detail::histogram::max_privatized_dynamic_smem_bins(sm100_policy, 4, 1) == 57088);
  STATIC_REQUIRE(cub::detail::histogram::max_privatized_dynamic_smem_bins(sm100_policy, 4, 4) == 14272);
  STATIC_REQUIRE(sm100_policy.dynamic_smem_range_max_bins == 2048);
  STATIC_REQUIRE(sm100_policy.dynamic_smem_even_2ch_max_bins == 28544);
  STATIC_REQUIRE(sm100_policy.dynamic_smem_even_3ch_max_bins == 19029);
  STATIC_REQUIRE(sm100_policy.dynamic_smem_even_4ch_max_bins == 8192);
  STATIC_REQUIRE(sm100_policy.gmem_threads_per_block == 768);
  STATIC_REQUIRE(sm100_policy.gmem_items_per_thread == 12);
  STATIC_REQUIRE(sm100_policy.static_smem_threads_per_block == sm100_policy.gmem_threads_per_block);

  constexpr auto sm100_range_u64_policy =
    cub::detail::histogram::policy_selector_from_types<long long, unsigned int, 1, 1, false>{}(
      cuda::compute_capability{10, 0});
  STATIC_REQUIRE(sm100_range_u64_policy.gmem_threads_per_block == 768);
  STATIC_REQUIRE(sm100_range_u64_policy.gmem_items_per_thread == 6);
  STATIC_REQUIRE(sm100_range_u64_policy.static_smem_threads_per_block == 384);
  STATIC_REQUIRE(sm100_range_u64_policy.static_smem_items_per_thread == 8);
  STATIC_REQUIRE(sm100_range_u64_policy.static_smem_min_blocks_per_sm == 3);

  constexpr auto sm100_multi_range_policy =
    cub::detail::histogram::policy_selector_from_types<int, unsigned int, 4, 3, false>{}(
      cuda::compute_capability{10, 0});
  STATIC_REQUIRE(sm100_multi_range_policy.gmem_threads_per_block == 1024);
  STATIC_REQUIRE(sm100_multi_range_policy.gmem_items_per_thread == 5);
  STATIC_REQUIRE(sm100_multi_range_policy.static_smem_threads_per_block == 384);
  STATIC_REQUIRE(sm100_multi_range_policy.static_smem_items_per_thread == 5);
  STATIC_REQUIRE(sm100_multi_range_policy.static_smem_min_blocks_per_sm == 3);

  STATIC_REQUIRE(cub::detail::histogram::should_use_dynamic_smem<false>(sm100_policy, 57088, 4, 1));
  STATIC_REQUIRE_FALSE(cub::detail::histogram::should_use_dynamic_smem<false>(sm100_policy, 57089, 4, 1));
  STATIC_REQUIRE(cub::detail::histogram::should_use_dynamic_smem<false>(sm100_policy, 2048, 4, 3));
  STATIC_REQUIRE_FALSE(cub::detail::histogram::should_use_dynamic_smem<false>(sm100_policy, 2049, 4, 3));
  STATIC_REQUIRE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 8192, 4, 4));
  STATIC_REQUIRE_FALSE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 8193, 4, 4));
  STATIC_REQUIRE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 28544, 4, 2));
  STATIC_REQUIRE_FALSE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 28545, 4, 2));
  STATIC_REQUIRE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 19029, 4, 3));
  STATIC_REQUIRE_FALSE(cub::detail::histogram::should_use_dynamic_smem<true>(sm100_policy, 19030, 4, 3));

  using max_policy_t = typename cub::detail::histogram::policy_hub<int, unsigned int, 1, 1, true>::MaxPolicy;
  const auto legacy_policy =
    cub::detail::histogram::policy_selector_from_hub<max_policy_t>{}(cuda::compute_capability{10, 0});
  REQUIRE(legacy_policy.static_smem_max_privatized_bytes == 256 * sizeof(unsigned int));
  REQUIRE(legacy_policy.dynamic_smem_max_privatized_bytes == 0);
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
