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

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceHistogram::HistogramEven works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramEven works with user provided memory and environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramRange works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramRange works with user provided memory and environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramEven works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramRange works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramEven 2D works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramRange 2D works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramEven 2D works with default environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramRange 2D works with default environment", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::HistogramEven uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramEven uses custom stream", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::HistogramRange uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramRange uses custom stream", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::MultiHistogramEven uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramEven uses custom stream", "[histogram][device]")
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
C2H_TEST("DeviceHistogram::MultiHistogramRange works with user provided memory and environment", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::MultiHistogramRange uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramRange uses custom stream", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::HistogramEven 2D uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramEven 2D uses custom stream", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::HistogramRange 2D uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::HistogramRange 2D uses custom stream", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::MultiHistogramEven 2D uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramEven 2D uses custom stream", "[histogram][device]")
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
TEST_CASE("DeviceHistogram::MultiHistogramEven works with user provided memory and environment", "[histogram][device]")
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

C2H_TEST("DeviceHistogram::MultiHistogramRange 2D uses environment", "[histogram][device]")
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

TEST_CASE("DeviceHistogram::MultiHistogramRange 2D uses custom stream", "[histogram][device]")
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
  _CCCL_API constexpr auto operator()(cuda::compute_capability) const -> cub::HistogramPolicy
  {
    return {BlockThreads, 1, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, false, cub::SMEM, false, 0};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

C2H_TEST("DeviceHistogram::HistogramEven can be tuned", "[histogram][device]", block_sizes)
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

C2H_TEST("DeviceHistogram::HistogramRange can be tuned", "[histogram][device]", block_sizes)
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

C2H_TEST("DeviceHistogram::MultiHistogramEven can be tuned", "[histogram][device]", block_sizes)
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

C2H_TEST("DeviceHistogram::MultiHistogramRange can be tuned", "[histogram][device]", block_sizes)
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
C2H_TEST("HistogramPolicy", "[histogram][device]")
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::HistogramPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::HistogramPolicy>);

  // aggregate init
  constexpr auto p1 = cub::HistogramPolicy{
    128, 7, 4, cub::BLOCK_LOAD_DIRECT, cub::CacheLoadModifier::LOAD_LDG, false, cub::SMEM, false, 2048};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::HistogramPolicy{
    .threads_per_block                = 128,
    .pixels_per_thread                = 7,
    .vec_size                         = 4,
    .load_algorithm                   = cub::BLOCK_LOAD_DIRECT,
    .load_modifier                    = cub::CacheLoadModifier::LOAD_LDG,
    .rle_compress                     = false,
    .mem_preference                   = cub::SMEM,
    .use_work_stealing                = false,
    .init_kernel_pdl_trigger_max_bins = 2048};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
