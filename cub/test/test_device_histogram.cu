/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Test of DeviceHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_histogram.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda/std/__cccl/dialect.h>
#include <cuda/std/array>
#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>
#include <typeinfo>

#include "c2h/vector.cuh"
#include "test_util.h"

#define TEST_HALF_T _CCCL_HAS_NVFP16

#if TEST_HALF_T
#  include <cuda_fp16.h>
#endif

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

bool g_verbose_input    = false;
bool g_verbose          = false;
int g_timing_iterations = 0;

template <typename It>
auto castIfHalfPointer(It it) -> It
{
  return it;
}

#if TEST_HALF_T
auto castIfHalfPointer(half_t* p) -> __half*
{
  return reinterpret_cast<__half*>(p);
}
#endif

template <typename T, std::size_t N>
auto toPtrArray(c2h::device_vector<T> (&in)[N]) -> ::cuda::std::array<T*, N>
{
  ::cuda::std::array<T*, N> r;
  for (std::size_t i = 0; i < N; i++)
  {
    r[i] = thrust::raw_pointer_cast(in[i].data());
  }
  return r;
}

//---------------------------------------------------------------------
// Dispatch to different DeviceHistogram entrypoints
//---------------------------------------------------------------------

template <int NUM_ACTIVE_CHANNELS,
          int NUM_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT>
void Even(
  int iterations,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples, ///< [in] The pointer to the multi-channel input sequence of data samples. The samples
                             ///< from different channels are assumed to be interleaved (e.g., an array of 32-bit
                             ///< pixels where each pixel consists of four RGBA 8-bit samples).
  c2h::device_vector<CounterT> (&d_histogram)[NUM_ACTIVE_CHANNELS],
  int* num_levels, ///< [in] The number of boundaries (levels) for delineating histogram samples in each active
                   ///< channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is
                   ///< <tt>num_levels[i]</tt> - 1.
  LevelT* lower_level, ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active
                       ///< channel.
  LevelT* upper_level, ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active
                       ///< channel.
  OffsetT num_row_pixels, ///< [in] The number of multi-channel pixels per row in the region of interest
  OffsetT num_rows, ///< [in] The number of rows in the region of interest
  OffsetT row_stride_bytes) ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
  for (int i = 0; i < iterations; ++i)
  {
    _CCCL_IF_CONSTEXPR (NUM_ACTIVE_CHANNELS == 1 && NUM_CHANNELS == 1)
    {
      const auto error = DeviceHistogram::HistogramEven(
        d_temp_storage,
        temp_storage_bytes,
        castIfHalfPointer(d_samples),
        toPtrArray(d_histogram)[0],
        num_levels[0],
        castIfHalfPointer(lower_level)[0],
        castIfHalfPointer(upper_level)[0],
        num_row_pixels,
        num_rows,
        row_stride_bytes);
      AssertEquals(error, cudaSuccess);
    }
    else
    {
      const auto error = DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        d_temp_storage,
        temp_storage_bytes,
        castIfHalfPointer(d_samples),
        toPtrArray(d_histogram).data(),
        num_levels,
        castIfHalfPointer(lower_level),
        castIfHalfPointer(upper_level),
        num_row_pixels,
        num_rows,
        row_stride_bytes);
      AssertEquals(error, cudaSuccess);
    }
  }
}

template <int NUM_ACTIVE_CHANNELS,
          int NUM_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT>
void Range(
  int iterations,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples, ///< [in] The pointer to the multi-channel input sequence of data samples. The samples
                             ///< from different channels are assumed to be interleaved (e.g., an array of 32-bit
                             ///< pixels where each pixel consists of four RGBA 8-bit samples).
  c2h::device_vector<CounterT> (&d_histogram)[NUM_ACTIVE_CHANNELS],
  int* num_levels, ///< [in] The number of boundaries (levels) for delineating histogram samples in each active
                   ///< channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is
                   ///< <tt>num_levels[i]</tt> - 1.
  c2h::device_vector<LevelT> (&d_levels)[NUM_ACTIVE_CHANNELS],
  OffsetT num_row_pixels, ///< [in] The number of multi-channel pixels per row in the region of interest
  OffsetT num_rows, ///< [in] The number of rows in the region of interest
  OffsetT row_stride_bytes) ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
  for (int i = 0; i < iterations; ++i)
  {
    _CCCL_IF_CONSTEXPR (NUM_ACTIVE_CHANNELS == 1 && NUM_CHANNELS == 1) // FIXME(bgruber): port to C++11
    {
      const auto error = DeviceHistogram::HistogramRange(
        d_temp_storage,
        temp_storage_bytes,
        castIfHalfPointer(d_samples),
        toPtrArray(d_histogram)[0],
        num_levels[0],
        castIfHalfPointer(toPtrArray(d_levels).data())[0],
        num_row_pixels,
        num_rows,
        row_stride_bytes);
      AssertEquals(error, cudaSuccess);
    }
    else
    {
      const auto error = DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        d_temp_storage,
        temp_storage_bytes,
        castIfHalfPointer(d_samples),
        toPtrArray(d_histogram).data(),
        num_levels,
        castIfHalfPointer(toPtrArray(d_levels).data()),
        num_row_pixels,
        num_rows,
        row_stride_bytes);
      AssertEquals(error, cudaSuccess);
    }
  }
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

template <typename T, typename LevelT>
auto GenerateSample(LevelT max_level, int entropy_reduction) -> T
{
  unsigned int bits;
  RandomBits(bits, entropy_reduction);
  const auto max = std::numeric_limits<unsigned int>::max();
  return static_cast<T>(static_cast<float>(bits) / static_cast<float>(max) * max_level); // TODO(bgruber): this could
                                                                                         // use a better generator
}

template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename LevelT, typename OffsetT>
auto CreateHistogramSamples(
  LevelT max_level, int entropy_reduction, OffsetT num_row_pixels, OffsetT num_rows, OffsetT row_stride_bytes)
  -> c2h::host_vector<SampleT>
{
  AssertTrue(row_stride_bytes % sizeof(SampleT) == 0);
  c2h::host_vector<SampleT> h_samples(num_rows * (row_stride_bytes / sizeof(SampleT)));

  // Initialize samples
  for (OffsetT row = 0; row < num_rows; ++row)
  {
    for (OffsetT pixel = 0; pixel < num_row_pixels; ++pixel)
    {
      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        // TODO(bgruber): use an mdspan to access h_samples
        // Sample offset
        OffsetT offset = row * (row_stride_bytes / sizeof(SampleT)) + pixel * NUM_CHANNELS + channel;

        // Init sample value
        h_samples[offset] = GenerateSample<SampleT>(max_level, entropy_reduction);
        if (g_verbose_input)
        {
          if (channel > 0)
          {
            printf(", ");
          }
          std::cout << CoutCast(h_samples[offset]);
        }
      }
    }
  }
  return h_samples;
}

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename CounterT,
          typename SampleIteratorT,
          typename TransformOp,
          typename OffsetT>
auto InitializeBinsWithSolution(
  SampleIteratorT h_samples,
  TransformOp&& transform_op,
  int* num_levels,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_bytes) -> cuda::std::array<c2h::host_vector<CounterT>, NUM_ACTIVE_CHANNELS>
{
  auto h_histogram = cuda::std::array<c2h::host_vector<CounterT>, NUM_ACTIVE_CHANNELS>{};
  for (int c = 0; c < NUM_ACTIVE_CHANNELS; ++c)
  {
    h_histogram[c].resize(num_levels[c] - 1);
  }

  // Initialize samples
  if (g_verbose_input)
  {
    printf("Samples: \n");
  }
  for (OffsetT row = 0; row < num_rows; ++row)
  {
    for (OffsetT pixel = 0; pixel < num_row_pixels; ++pixel)
    {
      if (g_verbose_input)
      {
        printf("[");
      }
      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        // Sample offset
        using SampleT = typename std::iterator_traits<SampleIteratorT>::value_type;
        // TODO(bgruber): use an mdspan to access h_samples
        OffsetT offset = row * (row_stride_bytes / sizeof(SampleT)) + pixel * NUM_CHANNELS + channel;

        // Update sample bin
        const int bin = transform_op(channel, h_samples[offset]);
        if (g_verbose_input)
        {
          printf(" (%d)", bin);
        }
        fflush(stdout);
        if (bin >= 0 && bin < static_cast<int>(h_histogram[channel].size()))
        {
          // valid bin
          ++h_histogram[channel][bin];
        }
      }
      if (g_verbose_input)
      {
        printf("]");
      }
    }
    if (g_verbose_input)
    {
      printf("\n\n");
    }
  }
  return h_histogram;
}

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename SampleIteratorT>
void TestHistogramEven(
  LevelT max_level,
  int entropy_reduction,
  int* num_levels, ///< [in] The number of boundaries (levels) for delineating histogram samples in
                   ///< each active channel.  Implies that the number of bins for
                   ///< channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
  LevelT* lower_level, ///< [in] The lower sample value bound (inclusive) for the lowest histogram
                       ///< bin in each active channel.
  LevelT* upper_level, ///< [in] The upper sample value bound (exclusive) for the highest histogram
                       ///< bin in each active channel.
  OffsetT num_row_pixels, ///< [in] The number of multi-channel pixels per row in the region of interest
  OffsetT num_rows, ///< [in] The number of rows in the region of interest
  OffsetT row_stride_bytes, ///< [in] The number of bytes between starts of consecutive rows in the region of interest
  SampleIteratorT h_samples,
  SampleIteratorT d_samples)
{
  const OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

  printf("\n----------------------------\n");
  printf(
    "%s cub::DeviceHistogram::Even (%s) "
    "%d pixels (%d height, %d width, %d-byte row stride), "
    "%d %d-byte %s samples (entropy reduction %d), "
    "%s levels, %s counters, %d/%d channels, max sample ",
    "CUB",
    (std::is_pointer<SampleIteratorT>::value) ? "pointer" : "iterator",
    (int) (num_row_pixels * num_rows),
    (int) num_rows,
    (int) num_row_pixels,
    (int) row_stride_bytes,
    (int) total_samples,
    (int) sizeof(SampleT),
    typeid(SampleT).name(),
    entropy_reduction,
    typeid(LevelT).name(),
    typeid(CounterT).name(),
    NUM_ACTIVE_CHANNELS,
    NUM_CHANNELS);
  std::cout << CoutCast(max_level) << "\n";
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    std::cout << "\tChannel " << channel << ": " << num_levels[channel] - 1 << " bins "
              << "[" << lower_level[channel] << ", " << upper_level[channel] << ")\n";
  }
  fflush(stdout);

  // Allocate and initialize host and device data
  LevelT fpScales[NUM_ACTIVE_CHANNELS];
  (void) fpScales; // TODO(bgruber): use [[maybe_unsued]] in C++17

  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    _CCCL_IF_CONSTEXPR (!std::is_integral<LevelT>::value)
    {
      fpScales[channel] =
        LevelT{1}
        / static_cast<LevelT>(
          (upper_level[channel] - lower_level[channel]) / static_cast<LevelT>(num_levels[channel] - 1));
    }
  }

  auto transform_op = [&](int channel, SampleT sample) {
    const auto n   = num_levels[channel];
    const auto max = upper_level[channel];
    const auto min = lower_level[channel];
    if (sample < min || sample >= max)
    {
      return n; // Sample out of range
    }

    _CCCL_IF_CONSTEXPR (std::is_integral<LevelT>::value)
    {
      // Accurate bin computation following the arithmetic we guarantee in the HistoEven docs
      return static_cast<int>(
        static_cast<uint64_t>(sample - min) * static_cast<uint64_t>(n - 1) / static_cast<uint64_t>(max - min));
    }
    else
    {
      return static_cast<int>((static_cast<float>(sample) - min) * fpScales[channel]);
    }
    _LIBCUDACXX_UNREACHABLE();
  };
  auto h_histogram = InitializeBinsWithSolution<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT>(
    h_samples, transform_op, num_levels, num_row_pixels, num_rows, row_stride_bytes);

  // Allocate and initialize device data
  c2h::device_vector<CounterT> d_histogram[NUM_ACTIVE_CHANNELS];
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    d_histogram[channel].resize(num_levels[channel] - 1);
  }

  // Allocate temporary storage with "canary" zones
  size_t temp_storage_bytes = 0;
  Even<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    1,
    nullptr,
    temp_storage_bytes,
    d_samples,
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes);
  cuda::std::array<char, 256> canary_zone;
  constexpr char canary_token = 8;
  canary_zone.fill(canary_token);
  c2h::device_vector<char> d_temp_storage(temp_storage_bytes + canary_zone.size() * 2, canary_token);

  // Run warmup/correctness iteration
  Even<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    1,
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size(),
    temp_storage_bytes,
    d_samples,
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes);

  // Check canary zones
  if (g_verbose)
  {
    printf("Checking leading temp_storage canary zone (token = %d)\n"
           "------------------------------------------------------\n",
           static_cast<int>(canary_token));
  }
  int error = CompareDeviceResults(
    canary_zone.data(), thrust::raw_pointer_cast(d_temp_storage.data()), canary_zone.size(), true, g_verbose);
  AssertEquals(0, error);
  if (g_verbose)
  {
    printf("Checking trailing temp_storage canary zone (token = %d)\n"
           "-------------------------------------------------------\n",
           static_cast<int>(canary_token));
  }
  error = CompareDeviceResults(
    canary_zone.data(),
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size() + temp_storage_bytes,
    canary_zone.size(),
    true,
    g_verbose);
  AssertEquals(0, error);

  // Flush any stdout/stderr
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  fflush(stdout);
  fflush(stderr);

  // Check for correctness (and display results, if specified)
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    if (g_verbose)
    {
      printf("Checking histogram result (channel = %d)\n"
             "----------------------------------------\n",
             channel);
    }
    int channel_error = CompareDeviceResults(
      h_histogram[channel].data(),
      thrust::raw_pointer_cast(d_histogram[channel].data()),
      num_levels[channel] - 1,
      true,
      g_verbose);
    printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
    error |= channel_error;
  }

  // Performance
  GpuTimer gpu_timer;
  gpu_timer.Start();

  Even<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    g_timing_iterations,
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size(),
    temp_storage_bytes,
    d_samples,
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes);

  gpu_timer.Stop();
  float elapsed_millis = gpu_timer.ElapsedMillis();

  // Display performance
  if (g_timing_iterations > 0)
  {
    float avg_millis     = elapsed_millis / g_timing_iterations;
    float giga_rate      = float(total_samples) / avg_millis / 1000.0f / 1000.0f;
    float giga_bandwidth = giga_rate * sizeof(SampleT);
    printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
           avg_millis,
           giga_rate,
           giga_rate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
           giga_rate / NUM_CHANNELS,
           giga_bandwidth);
  }

  printf("\n\n");
  AssertEquals(0, error);
}

template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename CounterT, typename LevelT, typename OffsetT>
void TestHistogramRange(
  LevelT max_level,
  int entropy_reduction,
  int* num_levels, ///< [in] The number of boundaries (levels) for delineating
                   ///< histogram samples in each active channel.  Implies that the
                   ///< number of bins for channel<sub><em>i</em></sub> is
                   ///< <tt>num_levels[i]</tt> - 1.
  c2h::host_vector<LevelT>* levels, ///< [in] The lower sample value bound (inclusive) for the lowest
                                    ///< histogram bin in each active channel.
  OffsetT num_row_pixels, ///< [in] The number of multi-channel pixels per row in the region of interest
  OffsetT num_rows, ///< [in] The number of rows in the region of interest
  OffsetT row_stride_bytes) ///< [in] The number of bytes between starts of consecutive rows in the region
                            ///< of interest
{
  printf("\n----------------------------\n");
  printf(
    "%s cub::DeviceHistogram::Range %d pixels "
    "(%d height, %d width, %d-byte row stride), "
    "%d %d-byte %s samples (entropy reduction %d), "
    "%s levels, %s counters, %d/%d channels, max sample ",
    "CUB",
    (int) (num_row_pixels * num_rows),
    (int) num_rows,
    (int) num_row_pixels,
    (int) row_stride_bytes,
    (int) (num_rows * (row_stride_bytes / sizeof(SampleT))),
    (int) sizeof(SampleT),
    typeid(SampleT).name(),
    entropy_reduction,
    typeid(LevelT).name(),
    typeid(CounterT).name(),
    NUM_ACTIVE_CHANNELS,
    NUM_CHANNELS);
  std::cout << CoutCast(max_level) << "\n";
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    printf("Channel %d: %d bins", channel, num_levels[channel] - 1);
    if (g_verbose)
    {
      std::cout << "[ " << levels[channel][0];
      for (int level = 1; level < num_levels[channel]; ++level)
      {
        std::cout << ", " << levels[channel][level];
      }
      printf("]");
    }
    printf("\n");
  }
  fflush(stdout);

  // Allocate and initialize host and device data
  auto h_samples = CreateHistogramSamples<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT>(
    max_level, entropy_reduction, num_row_pixels, num_rows, row_stride_bytes);

  const auto transform_op = [&](int channel, SampleT sample) {
    // convert samples to bin-ids (num_levels is returned if sample is out of range)
    LevelT* l     = levels[channel].data();
    const auto n  = num_levels[channel];
    const int bin = static_cast<int>(std::upper_bound(l, l + n, static_cast<LevelT>(sample)) - l - 1);
    if (bin < 0)
    {
      return n; // Sample out of range
    }
    return bin;
  };
  auto h_histogram = InitializeBinsWithSolution<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT>(
    h_samples.data(), transform_op, num_levels, num_row_pixels, num_rows, row_stride_bytes);

  // Allocate and initialize device data
  c2h::device_vector<LevelT> d_levels[NUM_ACTIVE_CHANNELS];
  c2h::device_vector<CounterT> d_histogram[NUM_ACTIVE_CHANNELS];
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    d_histogram[channel].resize(num_levels[channel] - 1);
    d_levels[channel] = levels[channel];
  }
  auto d_samples = c2h::device_vector<SampleT>(h_samples);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  Range<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    1,
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes);

  // Allocate temporary storage with "canary" zones
  cuda::std::array<char, 256> canary_zone;
  constexpr char canary_token = 9;
  canary_zone.fill(canary_token);
  c2h::device_vector<char> d_temp_storage(temp_storage_bytes + canary_zone.size() * 2, canary_token);

  // Run warmup/correctness iteration
  Range<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    1,
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size(),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes);

  // Check canary zones
  int error = CompareDeviceResults(
    canary_zone.data(), thrust::raw_pointer_cast(d_temp_storage.data()), canary_zone.size(), true, g_verbose);
  AssertEquals(0, error);
  error = CompareDeviceResults(
    canary_zone.data(),
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size() + temp_storage_bytes,
    canary_zone.size(),
    true,
    g_verbose);
  AssertEquals(0, error);

  // Flush any stdout/stderr
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  fflush(stdout);
  fflush(stderr);

  // Check for correctness (and display results, if specified)
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    int channel_error = CompareDeviceResults(
      h_histogram[channel].data(),
      thrust::raw_pointer_cast(d_histogram[channel].data()),
      num_levels[channel] - 1,
      true,
      g_verbose);
    printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
    error |= channel_error;
  }

  // Performance
  GpuTimer gpu_timer;
  gpu_timer.Start();

  Range<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
    g_timing_iterations,
    thrust::raw_pointer_cast(d_temp_storage.data()) + canary_zone.size(),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histogram,
    num_levels,
    d_levels,
    num_row_pixels,
    num_rows,
    row_stride_bytes);

  gpu_timer.Stop();
  float elapsed_millis = gpu_timer.ElapsedMillis();

  // Display performance
  if (g_timing_iterations > 0)
  {
    float avg_millis     = elapsed_millis / g_timing_iterations;
    float giga_rate      = float(h_samples.size()) / avg_millis / 1000.0f / 1000.0f;
    float giga_bandwidth = giga_rate * sizeof(SampleT);
    printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
           avg_millis,
           giga_rate,
           giga_rate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
           giga_rate / NUM_CHANNELS,
           giga_bandwidth);
  }

  printf("\n\n");
  AssertEquals(0, error);
}

template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename CounterT, typename LevelT, typename OffsetT>
void TestHistogramEvenWithIteratorInput(
  LevelT max_level,
  int entropy_reduction,
  int* num_levels,
  LevelT* lower_level,
  LevelT* upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_bytes,
  std::false_type)
{
  ConstantInputIterator<SampleT> sample_itr(static_cast<SampleT>(lower_level[0]));
  TestHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
    max_level,
    entropy_reduction,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    sample_itr,
    sample_itr);
}

// We have to reinterpret cast `half_t *` to `__half *`, so testing with iterators is not supported.
template <int, int, typename..., typename... Ts>
void TestHistogramEvenWithIteratorInput(Ts&&...)
{}

template <typename SampleT, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename CounterT, typename LevelT, typename OffsetT>
void TestHistogramEvenVariations(
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_bytes,
  int entropy_reduction,
  int* num_levels,
  LevelT max_level,
  int max_num_levels)
{
  LevelT lower_level[NUM_ACTIVE_CHANNELS];
  LevelT upper_level[NUM_ACTIVE_CHANNELS];

  // Find smallest level increment
  const int max_bins               = max_num_levels - 1;
  const LevelT min_level_increment = max_level / static_cast<LevelT>(max_bins);

  // Set upper and lower levels for each channel
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    const int num_bins = num_levels[channel] - 1;
    lower_level[channel] =
      static_cast<LevelT>((max_level - (static_cast<LevelT>(num_bins) * min_level_increment)) / static_cast<LevelT>(2));
    upper_level[channel] =
      static_cast<LevelT>((max_level + (static_cast<LevelT>(num_bins) * min_level_increment)) / static_cast<LevelT>(2));
  }

  // pointer-based samples
  {
    auto h_samples = CreateHistogramSamples<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT>(
      max_level, entropy_reduction, num_row_pixels, num_rows, row_stride_bytes);
    auto d_samples = c2h::device_vector<SampleT>(h_samples);
    TestHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
      max_level,
      entropy_reduction,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_bytes,
      h_samples.data(),
      thrust::raw_pointer_cast(d_samples.data()));
  }

  // Test iterator-based samples
  TestHistogramEvenWithIteratorInput<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
    max_level,
    entropy_reduction,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_bytes,
    std::integral_constant<bool, std::is_same<SampleT, half_t>::value>{}); // TODO(bgruber): use if constexpr instead of
                                                                           // tag dispatch
}

template <typename SampleT, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename CounterT, typename LevelT, typename OffsetT>
void TestHistogramRange(
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_bytes,
  int entropy_reduction,
  int* num_levels,
  LevelT max_level,
  int max_num_levels)
{
  // Find smallest level increment
  const int max_bins               = max_num_levels - 1;
  const LevelT min_level_increment = max_level / static_cast<LevelT>(max_bins);

  c2h::host_vector<LevelT> levels[NUM_ACTIVE_CHANNELS];
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    levels[channel].resize(num_levels[channel]);

    const int num_bins = num_levels[channel] - 1;
    // FIXME(bgruber): qualifying lower_level with const breaks operator+ below. Bug?
    LevelT lower_level = (max_level - static_cast<LevelT>(num_bins * min_level_increment)) / static_cast<LevelT>(2);

    for (int level = 0; level < num_levels[channel]; ++level)
    {
      levels[channel][level] = lower_level + static_cast<LevelT>(level * min_level_increment);
    }
  }

  TestHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
    max_level, entropy_reduction, num_levels, levels, num_row_pixels, num_rows, row_stride_bytes);
}

template <typename SampleT, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename CounterT, typename LevelT, typename OffsetT>
void TestProblemSizes(LevelT max_level, int max_num_levels)
{
  // Test corner cases and sample a few different aspect ratios sizes
  for (const auto& p :
       std::vector<std::pair<OffsetT, OffsetT>>{{1920, 0}, {0, 0}, {15, 1}, {1920, 1080}, {1, 1}, {1000, 1}, {1, 1000}})
  {
    // TODO(bgruber): use structured bindings in C++17
    const OffsetT num_row_pixels = p.first;
    const OffsetT num_rows       = p.second;

    const OffsetT row_stride_bytes = num_row_pixels * NUM_CHANNELS * sizeof(SampleT);
    for (std::size_t padding : {std::size_t{0}, 13 * sizeof(SampleT)})
    {
      const OffsetT padded_row_stride_bytes = row_stride_bytes + static_cast<OffsetT>(padding);
      for (int entropy_reduction : {-1, 0, 5}) // entropy_reduction = -1 -> all samples == 0
      {
        int num_levels[NUM_ACTIVE_CHANNELS]{max_num_levels};
        for (int c = 1; c < NUM_ACTIVE_CHANNELS; ++c)
        {
          num_levels[c] = num_levels[c - 1] / 2 + 1;
        }

        TestHistogramEvenVariations<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
          num_row_pixels, num_rows, padded_row_stride_bytes, entropy_reduction, num_levels, max_level, max_num_levels);
        TestHistogramRange<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
          num_row_pixels, num_rows, padded_row_stride_bytes, entropy_reduction, num_levels, max_level, max_num_levels);
      }
    }
  }
}

template <typename SampleT, typename CounterT, typename LevelT, typename OffsetT>
void TestExtraChannels(LevelT max_level, int max_num_levels, std::true_type)
{
  TestProblemSizes<SampleT, 3, 3, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  TestProblemSizes<SampleT, 4, 4, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
}

template <typename...>
void TestExtraChannels(...)
{}

template <typename SampleT, typename CounterT, typename LevelT, typename OffsetT, bool TTestExtraChannels>
void TestChannels(LevelT max_level, int max_num_levels, std::true_type = {})
{
  TestProblemSizes<SampleT, 1, 1, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  TestProblemSizes<SampleT, 4, 3, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  // TODO(bgruber): use if constexpr in C++17
  TestExtraChannels<SampleT, CounterT, LevelT, OffsetT>(
    max_level, max_num_levels, std::integral_constant<bool, TTestExtraChannels>{});
}

template <typename...>
void TestChannels(...)
{}

void TestLevelsAliasing()
{
  constexpr int num_levels = 7;

  constexpr int h_samples[]{
    0,  2,  4,  6,  8,  10, 12, // levels
    1, // bin 0
    3,  3, // bin 1
    5,  5,  5, // bin 2
    7,  7,  7,  7, // bin 3
    9,  9,  9,  9,  9, // bin 4
    11, 11, 11, 11, 11, 11 // bin 5
  };

  auto d_histogram = c2h::device_vector<int>(num_levels - 1);
  auto d_samples   = c2h::device_vector<int>(std::begin(h_samples), std::end(h_samples));

  // Alias levels with samples (fancy way to `d_histogram[bin]++`).
  int* d_levels = thrust::raw_pointer_cast(d_samples.data());

  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceHistogram::HistogramRange(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    d_levels,
    static_cast<int>(d_samples.size())));
  auto d_temp_storage = c2h::device_vector<char>(temp_storage_bytes);

  CubDebugExit(cub::DeviceHistogram::HistogramRange(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    d_levels,
    static_cast<int>(d_samples.size())));

  auto h_histogram = c2h::host_vector<int>(d_histogram);
  for (int bin = 0; bin < num_levels - 1; bin++)
  {
    // Each bin should contain `bin + 1` samples. Since samples also contain
    // levels, they contribute one extra item to each bin.
    AssertEquals(bin + 2, h_histogram[bin]);
  }
}

// Regression test for NVIDIA/cub#489: integer rounding errors lead to incorrect
// bin detection:
void TestIntegerBinCalcs()
{
  constexpr int num_levels = 8;
  constexpr int num_bins   = num_levels - 1;

  constexpr int h_histogram_ref[num_bins]{1, 5, 0, 2, 1, 0, 0};
  constexpr int h_samples[]{2, 6, 7, 2, 3, 0, 2, 2, 6, 999};
  constexpr int lower_level = 0;
  constexpr int upper_level = 12;

  auto d_histogram = c2h::device_vector<int>(num_levels - 1);
  auto d_samples   = c2h::device_vector<int>(std::begin(h_samples), std::end(h_samples));

  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceHistogram::HistogramEven(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    static_cast<int>(d_samples.size())));
  auto d_temp_storage = c2h::device_vector<char>(temp_storage_bytes);

  CubDebugExit(cub::DeviceHistogram::HistogramEven(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    static_cast<int>(d_samples.size())));

  auto h_histogram = c2h::host_vector<int>(d_histogram);
  for (int bin = 0; bin < num_bins; ++bin)
  {
    AssertEquals(h_histogram_ref[bin], h_histogram[bin]);
  }
}

/**
 * @brief Our bin computation for HistogramEven is guaranteed only for when (max_level - min_level)
 * * num_bins does not overflow when using uint64_t arithmetic. In case bin computation could
 * overflow, we expect cudaErrorInvalidValue to be returned.
 */
template <typename SampleT>
void TestOverflow()
{
  using CounterT                   = uint32_t;
  constexpr std::size_t test_cases = 2;

  // Test data common across tests
  SampleT lower_level = 0;
  SampleT upper_level = ::cuda::std::numeric_limits<SampleT>::max();
  thrust::counting_iterator<SampleT> d_samples{0UL};
  thrust::device_vector<CounterT> d_histo_out(1024);
  CounterT* d_histogram = thrust::raw_pointer_cast(d_histo_out.data());
  const int num_samples = 1000;

  // Prepare per-test specific data
  constexpr std::size_t canary_bytes = 3;
  std::array<std::size_t, test_cases> temp_storage_bytes{canary_bytes, canary_bytes};
  std::array<int, test_cases> num_bins{1, 2};
  // Since test #1 is just a single bin, we expect it to succeed
  // Since we promote up to 64-bit integer arithmetic we expect tests to not overflow for types of
  // up to 4 bytes. For 64-bit and wider types, we do not perform further promotion to even wider
  // types, hence we expect cudaErrorInvalidValue to be returned to indicate of a potential overflow
  std::array<cudaError_t, test_cases> expected_status{
    cudaSuccess, sizeof(SampleT) <= 4UL ? cudaSuccess : cudaErrorInvalidValue};

  // Verify we always initializes temp_storage_bytes
  cudaError_t error{cudaSuccess};
  for (std::size_t i = 0; i < test_cases; i++)
  {
    error = cub::DeviceHistogram::HistogramEven(
      nullptr, temp_storage_bytes[i], d_samples, d_histogram, num_bins[i] + 1, lower_level, upper_level, num_samples);

    // Ensure that temp_storage_bytes has been initialized even in the presence of error
    AssertTrue(temp_storage_bytes[i] != canary_bytes);
  }

  // Allocate sufficient temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(std::max(temp_storage_bytes[0], temp_storage_bytes[1]));

  for (std::size_t i = 0; i < test_cases; i++)
  {
    error = cub::DeviceHistogram::HistogramEven(
      thrust::raw_pointer_cast(temp_storage.data()),
      temp_storage_bytes[i],
      d_samples,
      d_histogram,
      num_bins[i] + 1,
      lower_level,
      upper_level,
      num_samples);

    // Ensure we do not return an error on querying temporary storage requirements
    AssertEquals(error, expected_status[i]);
  }
}

int main(int argc, char* argv[])
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose       = args.CheckCmdLineFlag("v");
  g_verbose_input = args.CheckCmdLineFlag("v2");

  args.GetCmdLineArgument("i", g_timing_iterations);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--i=<timing iterations>] "
           "[--device=<device-id>] "
           "[--v] "
           "[--v2] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  TestOverflow<uint8_t>();
  TestOverflow<uint16_t>();
  TestOverflow<uint32_t>();
  TestOverflow<uint64_t>();

  TestLevelsAliasing();
  TestIntegerBinCalcs(); // regression test for NVIDIA/cub#489

#if TEST_HALF_T
  TestChannels<half_t, int, half_t, int, true>(half_t{256}, 256 + 1);
#endif

  TestChannels<signed char, int, int, int, true>(256, 256 + 1);
  TestChannels<unsigned short, int, int, int, false>(8192, 8192 + 1);

  // Make sure bin computation works fine when using int32 arithmetic
  TestChannels<unsigned short, int, unsigned short, int, false>(
    std::numeric_limits<unsigned short>::max(), std::numeric_limits<unsigned short>::max() + 1);
  // Make sure bin computation works fine when requiring int64 arithmetic
  TestChannels<unsigned int, int, unsigned int, int, false>(std::numeric_limits<unsigned int>::max(), 8192 + 1);
  TestChannels<float, int, float, int, false>(1.0, 256 + 1);

  // float samples, int levels, regression test for NVIDIA/cub#479.
  TestChannels<float, int, int, int, true>(12, 7);

  // Test down-conversion of size_t offsets to int
  TestChannels<unsigned char, int, int, long long, false>(
    256, 256 + 1, std::integral_constant<bool, sizeof(size_t) != sizeof(int)>{});

  return 0;
}
