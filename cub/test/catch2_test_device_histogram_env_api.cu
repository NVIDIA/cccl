// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_histogram.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/array>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceHistogram::HistogramEven accepts env with stream", "[histogram][env]")
{
  // example-begin histogram-even-env
  auto d_samples   = thrust::device_vector<int>{0, 2, 1, 0, 3, 4, 2, 1};
  int num_samples  = static_cast<int>(d_samples.size());
  int num_levels   = 6;
  int lower_level  = 0;
  int upper_level  = 5;
  auto d_histogram = thrust::device_vector<int>(num_levels - 1, 0);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::HistogramEven(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_samples,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::HistogramEven failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{2, 2, 2, 1, 1};
  // example-end histogram-even-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram == expected);
}

C2H_TEST("cub::DeviceHistogram::HistogramEven accepts env with stream (2D)", "[histogram][env]")
{
  // example-begin histogram-even-2d-env
  // 2D region of interest: 2 rows, 3 samples per row, row stride includes 1 padding element
  // Row 0: [0, 1, 2, PAD]   Row 1: [1, 2, 0, PAD]
  auto d_samples          = thrust::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  int num_levels          = 4; // 3 bins: [0,1), [1,2), [2,3)
  int lower_level         = 0;
  int upper_level         = 3;
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);

  auto d_histogram = thrust::device_vector<int>(num_levels - 1, 0);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::HistogramEven(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::HistogramEven (2D) failed with status: " << error << std::endl;
  }

  // Samples: 0,1,2, 1,2,0 → bin[0]=2, bin[1]=2, bin[2]=2
  thrust::device_vector<int> expected{2, 2, 2};
  // example-end histogram-even-2d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram == expected);
}

C2H_TEST("cub::DeviceHistogram::HistogramRange accepts env with stream", "[histogram][env]")
{
  // example-begin histogram-range-env
  auto d_samples   = thrust::device_vector<float>{2.2f, 6.1f, 7.5f, 2.9f, 3.5f, 0.3f, 2.9f, 2.1f};
  int num_samples  = static_cast<int>(d_samples.size());
  auto d_levels    = thrust::device_vector<float>{0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  int num_levels   = static_cast<int>(d_levels.size());
  auto d_histogram = thrust::device_vector<int>(num_levels - 1, 0);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::HistogramRange(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_samples,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::HistogramRange failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 5, 0, 2};
  // example-end histogram-range-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram == expected);
}

C2H_TEST("cub::DeviceHistogram::HistogramRange accepts env with stream (2D)", "[histogram][env]")
{
  // example-begin histogram-range-2d-env
  // 2D region of interest: 2 rows, 3 samples per row, row stride includes 1 padding element
  // Row 0: [0, 1, 2, PAD]   Row 1: [1, 2, 0, PAD]
  auto d_samples          = thrust::device_vector<int>{0, 1, 2, -1, 1, 2, 0, -1};
  auto d_levels           = thrust::device_vector<int>{0, 1, 2, 3}; // 3 bins: [0,1), [1,2), [2,3)
  int num_levels          = static_cast<int>(d_levels.size());
  int num_row_samples     = 3;
  int num_rows            = 2;
  size_t row_stride_bytes = 4 * sizeof(int);

  auto d_histogram = thrust::device_vector<int>(num_levels - 1, 0);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::HistogramRange(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_levels.data()),
    num_row_samples,
    num_rows,
    row_stride_bytes,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::HistogramRange (2D) failed with status: " << error << std::endl;
  }

  // Samples: 0,1,2, 1,2,0 → bin[0]=2, bin[1]=2, bin[2]=2
  thrust::device_vector<int> expected{2, 2, 2};
  // example-end histogram-range-2d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram == expected);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramEven accepts env with stream (1D)", "[histogram][env]")
{
  // example-begin multi-histogram-even-1d-env
  // 4-channel RGBA pixels, histogram 3 active channels
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // clang-format off
  // 2 pixels: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128)
  auto d_samples = thrust::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  // clang-format on
  int num_pixels = 2;

  // 5 levels per channel → 4 bins per channel: [0,1), [1,2), [2,3), [3,4)
  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = thrust::device_vector<int>(4, 0);
  auto d_histogram_g = thrust::device_vector<int>(4, 0);
  auto d_histogram_b = thrust::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, lower_level, upper_level, num_pixels, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::MultiHistogramEven failed with status: " << error << std::endl;
  }

  // R: 0→bin[0], 3→bin[3]
  thrust::device_vector<int> expected_r{1, 0, 0, 1};
  // G: 2→bin[2], 4→out of range
  thrust::device_vector<int> expected_g{0, 0, 1, 0};
  // B: 1→bin[1], 2→bin[2]
  thrust::device_vector<int> expected_b{0, 1, 1, 0};
  // example-end multi-histogram-even-1d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramEven accepts env with stream (2D)", "[histogram][env]")
{
  // example-begin multi-histogram-even-2d-env
  // 4-channel RGBA pixels, histogram 3 active channels, 2D region
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 rows, 2 pixels per row, stride includes 1 extra padding pixel per row
  // Row 0: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128), (PAD, PAD, PAD, PAD)
  // Row 1: (R=1, G=1, B=3, A=200), (R=2, G=3, B=0, A=100), (PAD, PAD, PAD, PAD)
  auto d_samples = thrust::device_vector<unsigned char>{
    0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char); // 3 pixels wide, 2 used

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels            = {5, 5, 5};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> lower_level = {0, 0, 0};
  cuda::std::array<unsigned char, NUM_ACTIVE_CHANNELS> upper_level = {4, 4, 4};

  auto d_histogram_r = thrust::device_vector<int>(4, 0);
  auto d_histogram_g = thrust::device_vector<int>(4, 0);
  auto d_histogram_b = thrust::device_vector<int>(4, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::MultiHistogramEven (2D) failed with status: " << error << std::endl;
  }

  // R: 0, 3, 1, 2 → bin[0]=1, bin[1]=1, bin[2]=1, bin[3]=1
  thrust::device_vector<int> expected_r{1, 1, 1, 1};
  // G: 2, 4, 1, 3 → bin[1]=1, bin[2]=1, bin[3]=1 (4 is out of range)
  thrust::device_vector<int> expected_g{0, 1, 1, 1};
  // B: 1, 2, 3, 0 → bin[0]=1, bin[1]=1, bin[2]=1, bin[3]=1
  thrust::device_vector<int> expected_b{1, 1, 1, 1};
  // example-end multi-histogram-even-2d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramRange accepts env with stream (1D)", "[histogram][env]")
{
  // example-begin multi-histogram-range-1d-env
  // 4-channel RGBA pixels, histogram 3 active channels
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 pixels: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128)
  auto d_samples = thrust::device_vector<unsigned char>{0, 2, 1, 255, 3, 4, 2, 128};
  int num_pixels = 2;

  // Custom bin boundaries per channel
  auto d_levels_r = thrust::device_vector<unsigned char>{0, 2, 4}; // 2 bins: [0,2), [2,4)
  auto d_levels_g = thrust::device_vector<unsigned char>{0, 3, 5}; // 2 bins: [0,3), [3,5)
  auto d_levels_b = thrust::device_vector<unsigned char>{0, 1, 2, 3}; // 3 bins: [0,1), [1,2), [2,3)

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = thrust::device_vector<int>(2, 0);
  auto d_histogram_g = thrust::device_vector<int>(2, 0);
  auto d_histogram_b = thrust::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()), d_histogram, num_levels, d_levels, num_pixels, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::MultiHistogramRange failed with status: " << error << std::endl;
  }

  // R: 0→[0,2), 3→[2,4)
  thrust::device_vector<int> expected_r{1, 1};
  // G: 2→[0,3), 4→[3,5)
  thrust::device_vector<int> expected_g{1, 1};
  // B: 1→[1,2), 2→[2,3)
  thrust::device_vector<int> expected_b{0, 1, 1};
  // example-end multi-histogram-range-1d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramRange accepts env with stream (2D)", "[histogram][env]")
{
  // example-begin multi-histogram-range-2d-env
  // 4-channel RGBA pixels, histogram 3 active channels, 2D region
  [[maybe_unused]] constexpr int NUM_CHANNELS        = 4;
  [[maybe_unused]] constexpr int NUM_ACTIVE_CHANNELS = 3;

  // 2 rows, 2 pixels per row, stride includes 1 extra padding pixel per row
  // Row 0: (R=0, G=2, B=1, A=255), (R=3, G=4, B=2, A=128), (PAD, PAD, PAD, PAD)
  // Row 1: (R=1, G=1, B=3, A=200), (R=2, G=3, B=0, A=100), (PAD, PAD, PAD, PAD)
  auto d_samples = thrust::device_vector<unsigned char>{
    0, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0};

  int num_row_pixels      = 2;
  int num_rows            = 2;
  size_t row_stride_bytes = 3 * NUM_CHANNELS * sizeof(unsigned char); // 3 pixels wide, 2 used

  auto d_levels_r = thrust::device_vector<unsigned char>{0, 2, 4}; // 2 bins: [0,2), [2,4)
  auto d_levels_g = thrust::device_vector<unsigned char>{0, 3, 5}; // 2 bins: [0,3), [3,5)
  auto d_levels_b = thrust::device_vector<unsigned char>{0, 1, 2, 3}; // 3 bins: [0,1), [1,2), [2,3)

  cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_levels = {3, 3, 4};

  cuda::std::array<const unsigned char*, NUM_ACTIVE_CHANNELS> d_levels = {
    thrust::raw_pointer_cast(d_levels_r.data()),
    thrust::raw_pointer_cast(d_levels_g.data()),
    thrust::raw_pointer_cast(d_levels_b.data())};

  auto d_histogram_r = thrust::device_vector<int>(2, 0);
  auto d_histogram_g = thrust::device_vector<int>(2, 0);
  auto d_histogram_b = thrust::device_vector<int>(3, 0);

  cuda::std::array<int*, NUM_ACTIVE_CHANNELS> d_histogram = {
    thrust::raw_pointer_cast(d_histogram_r.data()),
    thrust::raw_pointer_cast(d_histogram_g.data()),
    thrust::raw_pointer_cast(d_histogram_b.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceHistogram::MultiHistogramRange (2D) failed with status: " << error << std::endl;
  }

  // R: 0, 3, 1, 2 → [0,2)=2, [2,4)=2
  thrust::device_vector<int> expected_r{2, 2};
  // G: 2, 4, 1, 3 → [0,3)=2, [3,5)=2
  thrust::device_vector<int> expected_g{2, 2};
  // B: 1, 2, 3, 0 → [0,1)=1, [1,2)=1, [2,3)=1 (3 is out of range)
  thrust::device_vector<int> expected_b{1, 1, 1};
  // example-end multi-histogram-range-2d-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_histogram_r == expected_r);
  REQUIRE(d_histogram_g == expected_g);
  REQUIRE(d_histogram_b == expected_b);
}
