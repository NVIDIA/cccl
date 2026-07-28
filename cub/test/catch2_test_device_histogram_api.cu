// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_histogram.cuh>

#include <thrust/device_vector.h>

#include <cuda/std/array>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceHistogram::HistogramEven non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
    nullptr, temp_storage_bytes, samples.begin(), thrust::raw_pointer_cast(histogram.data()), 2, 0, 10, 1);
}

C2H_TEST("cub::DeviceHistogram::HistogramEven 2D non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
    nullptr,
    temp_storage_bytes,
    samples.begin(),
    thrust::raw_pointer_cast(histogram.data()),
    2,
    0,
    10,
    1,
    1,
    sizeof(int));
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramEven non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  ::cuda::std::array<int*, 1> d_histogram{thrust::raw_pointer_cast(histogram.data())};
  ::cuda::std::array<int, 1> num_levels{2};
  ::cuda::std::array<int, 1> lower_level{0};
  ::cuda::std::array<int, 1> upper_level{10};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<1, 1>(
    nullptr, temp_storage_bytes, samples.begin(), d_histogram, num_levels, lower_level, upper_level, 1);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramEven 2D non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  ::cuda::std::array<int*, 1> d_histogram{thrust::raw_pointer_cast(histogram.data())};
  ::cuda::std::array<int, 1> num_levels{2};
  ::cuda::std::array<int, 1> lower_level{0};
  ::cuda::std::array<int, 1> upper_level{10};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<1, 1>(
    nullptr, temp_storage_bytes, samples.begin(), d_histogram, num_levels, lower_level, upper_level, 1, 1, sizeof(int));
}

C2H_TEST("cub::DeviceHistogram::HistogramRange non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  thrust::device_vector<int> levels{0, 5, 10};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(
    nullptr,
    temp_storage_bytes,
    samples.begin(),
    thrust::raw_pointer_cast(histogram.data()),
    3,
    thrust::raw_pointer_cast(levels.data()),
    1);
}

C2H_TEST("cub::DeviceHistogram::HistogramRange 2D non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  thrust::device_vector<int> levels{0, 5, 10};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(
    nullptr,
    temp_storage_bytes,
    samples.begin(),
    thrust::raw_pointer_cast(histogram.data()),
    3,
    thrust::raw_pointer_cast(levels.data()),
    1,
    1,
    sizeof(int));
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramRange non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  thrust::device_vector<int> levels{0, 5, 10};
  ::cuda::std::array<int*, 1> d_histogram{thrust::raw_pointer_cast(histogram.data())};
  ::cuda::std::array<int, 1> num_levels{3};
  ::cuda::std::array<const int*, 1> d_levels{thrust::raw_pointer_cast(levels.data())};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<1, 1>(
    nullptr, temp_storage_bytes, samples.begin(), d_histogram, num_levels, d_levels, 1);
}

C2H_TEST("cub::DeviceHistogram::MultiHistogramRange 2D non-env overload is not ambiguous", "[histogram][device]")
{
  thrust::device_vector<int> samples(1);
  thrust::device_vector<int> histogram(1);
  thrust::device_vector<int> levels{0, 5, 10};
  ::cuda::std::array<int*, 1> d_histogram{thrust::raw_pointer_cast(histogram.data())};
  ::cuda::std::array<int, 1> num_levels{3};
  ::cuda::std::array<const int*, 1> d_levels{thrust::raw_pointer_cast(levels.data())};
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<1, 1>(
    nullptr, temp_storage_bytes, samples.begin(), d_histogram, num_levels, d_levels, 1, 1, sizeof(int));
}
