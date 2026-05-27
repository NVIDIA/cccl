// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises the thread_local detection-stream / detection-buffer cache used by
// dispatch_range's uniform-levels detection path: sequential calls on multiple
// user streams, concurrent calls from multiple threads, and the cross-device
// hazard where the cache is bound to the device current at first call.

#include <cub/device/device_histogram.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace
{

using sample_t  = int;
using counter_t = int;

// Jittered uniform spacing keeps DispatchRange on the SearchTransform path,
// which is where the thread_local detection cache is exercised. Fixed seed
// makes the levels reproducible across runs.
auto make_levels(int num_bins) -> thrust::host_vector<sample_t>
{
  constexpr double upper = 1024.0;
  const double step      = upper / static_cast<double>(num_bins);
  thrust::host_vector<sample_t> levels(num_bins + 1);
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<double> jitter(-0.25, 0.25);
  levels[0]        = 0;
  levels[num_bins] = static_cast<sample_t>(upper);
  for (int i = 1; i < num_bins; ++i)
  {
    sample_t v = static_cast<sample_t>(i * step + step * jitter(rng));
    if (v <= levels[i - 1])
    {
      v = static_cast<sample_t>(levels[i - 1] + 1);
    }
    levels[i] = v;
  }
  if (levels[num_bins] <= levels[num_bins - 1])
  {
    levels[num_bins] = static_cast<sample_t>(levels[num_bins - 1] + 1);
  }
  return levels;
}

auto reference_histogram(const thrust::host_vector<sample_t>& samples, const thrust::host_vector<sample_t>& levels)
  -> thrust::host_vector<counter_t>
{
  const int num_bins = static_cast<int>(levels.size()) - 1;
  thrust::host_vector<counter_t> ref(num_bins, 0);
  for (sample_t s : samples)
  {
    auto ub = std::upper_bound(levels.begin(), levels.end(), s);
    if (ub == levels.begin() || ub == levels.end())
    {
      continue;
    }
    ++ref[std::distance(levels.begin(), ub) - 1];
  }
  return ref;
}

void run_histogram_range(
  cudaStream_t stream,
  const thrust::device_vector<sample_t>& d_samples,
  const thrust::device_vector<sample_t>& d_levels,
  thrust::device_vector<counter_t>& d_histogram)
{
  const int num_levels = static_cast<int>(d_levels.size());
  size_t temp_bytes    = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceHistogram::HistogramRange(
            nullptr,
            temp_bytes,
            thrust::raw_pointer_cast(d_samples.data()),
            thrust::raw_pointer_cast(d_histogram.data()),
            num_levels,
            thrust::raw_pointer_cast(d_levels.data()),
            static_cast<int>(d_samples.size()),
            stream));
  thrust::device_vector<unsigned char> temp(temp_bytes);
  REQUIRE(cudaSuccess
          == cub::DeviceHistogram::HistogramRange(
            thrust::raw_pointer_cast(temp.data()),
            temp_bytes,
            thrust::raw_pointer_cast(d_samples.data()),
            thrust::raw_pointer_cast(d_histogram.data()),
            num_levels,
            thrust::raw_pointer_cast(d_levels.data()),
            static_cast<int>(d_samples.size()),
            stream));
}

} // namespace

C2H_TEST("DeviceHistogram::HistogramRange thread_local cache: sequential calls on multiple user streams",
         "[histogram][device]")
{
  constexpr int num_bins    = 32;
  constexpr int num_samples = 4096;

  const auto h_levels = make_levels(num_bins);
  thrust::host_vector<sample_t> h_samples(num_samples);
  for (int i = 0; i < num_samples; ++i)
  {
    h_samples[i] = static_cast<sample_t>((i * 31) % 1024);
  }
  const auto h_ref = reference_histogram(h_samples, h_levels);

  thrust::device_vector<sample_t> d_samples = h_samples;
  thrust::device_vector<sample_t> d_levels  = h_levels;

  cudaStream_t stream_a;
  cudaStream_t stream_b;
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream_a));
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream_b));

  for (cudaStream_t s : {stream_a, stream_b, stream_a})
  {
    thrust::device_vector<counter_t> d_histogram(num_bins, 0);
    run_histogram_range(s, d_samples, d_levels, d_histogram);
    REQUIRE(cudaSuccess == cudaStreamSynchronize(s));
    thrust::host_vector<counter_t> h_got = d_histogram;
    CHECK(h_got == h_ref);
  }

  REQUIRE(cudaSuccess == cudaStreamDestroy(stream_a));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream_b));
}

C2H_TEST("DeviceHistogram::HistogramRange thread_local cache: concurrent threads on the same device",
         "[histogram][device]")
{
  constexpr int num_bins    = 64;
  constexpr int num_samples = 8192;
  constexpr int num_threads = 4;

  const auto h_levels = make_levels(num_bins);
  thrust::host_vector<sample_t> h_samples(num_samples);
  for (int i = 0; i < num_samples; ++i)
  {
    h_samples[i] = static_cast<sample_t>((i * 13 + 7) % 1024);
  }
  const auto h_ref = reference_histogram(h_samples, h_levels);

  thrust::device_vector<sample_t> d_samples = h_samples;
  thrust::device_vector<sample_t> d_levels  = h_levels;

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t)
  {
    threads.emplace_back([&]() {
      cudaStream_t stream;
      if (cudaStreamCreate(&stream) != cudaSuccess)
      {
        failures.fetch_add(1);
        return;
      }
      for (int it = 0; it < 2; ++it)
      {
        thrust::device_vector<counter_t> d_histogram(num_bins, 0);
        const int num_levels = static_cast<int>(d_levels.size());
        size_t temp_bytes    = 0;
        if (cub::DeviceHistogram::HistogramRange(
              nullptr,
              temp_bytes,
              thrust::raw_pointer_cast(d_samples.data()),
              thrust::raw_pointer_cast(d_histogram.data()),
              num_levels,
              thrust::raw_pointer_cast(d_levels.data()),
              static_cast<int>(d_samples.size()),
              stream)
            != cudaSuccess)
        {
          failures.fetch_add(1);
          continue;
        }
        thrust::device_vector<unsigned char> temp(temp_bytes);
        if (cub::DeviceHistogram::HistogramRange(
              thrust::raw_pointer_cast(temp.data()),
              temp_bytes,
              thrust::raw_pointer_cast(d_samples.data()),
              thrust::raw_pointer_cast(d_histogram.data()),
              num_levels,
              thrust::raw_pointer_cast(d_levels.data()),
              static_cast<int>(d_samples.size()),
              stream)
            != cudaSuccess)
        {
          failures.fetch_add(1);
          continue;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
          failures.fetch_add(1);
          continue;
        }
        thrust::host_vector<counter_t> h_got = d_histogram;
        if (h_got != h_ref)
        {
          failures.fetch_add(1);
        }
      }
      cudaStreamDestroy(stream);
    });
  }
  for (auto& th : threads)
  {
    th.join();
  }
  CHECK(failures.load() == 0);
}

C2H_TEST("DeviceHistogram::HistogramRange thread_local cache: same thread switches devices between calls",
         "[histogram][device]")
{
  int num_devices = 0;
  REQUIRE(cudaSuccess == cudaGetDeviceCount(&num_devices));
  if (num_devices < 2)
  {
    SKIP("Requires >= 2 CUDA devices");
  }

  constexpr int num_bins    = 32;
  constexpr int num_samples = 4096;

  const auto h_levels = make_levels(num_bins);
  thrust::host_vector<sample_t> h_samples(num_samples);
  for (int i = 0; i < num_samples; ++i)
  {
    h_samples[i] = static_cast<sample_t>((i * 17) % 1024);
  }
  const auto h_ref = reference_histogram(h_samples, h_levels);

  for (int dev : {0, 1, 0})
  {
    REQUIRE(cudaSuccess == cudaSetDevice(dev));
    thrust::device_vector<sample_t> d_samples = h_samples;
    thrust::device_vector<sample_t> d_levels  = h_levels;
    thrust::device_vector<counter_t> d_histogram(num_bins, 0);
    run_histogram_range(/*stream=*/0, d_samples, d_levels, d_histogram);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    thrust::host_vector<counter_t> h_got = d_histogram;
    INFO("device " << dev);
    CHECK(h_got == h_ref);
  }
  REQUIRE(cudaSuccess == cudaSetDevice(0));
}
