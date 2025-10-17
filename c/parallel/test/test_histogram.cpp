//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include <cuda_runtime.h>

#include "test_util.h"
#include <cccl/c/histogram.h>

using sample_types =
  c2h::type_list<std::int8_t,
                 std::uint16_t,
                 std::int32_t,
                 std::uint64_t,
#if _CCCL_HAS_NVFP16()
                 __half,
#endif
                 float,
                 double>;
using LevelT = double;

constexpr int num_channels        = 1;
constexpr int num_active_channels = 1;

void build_histogram(
  cccl_device_histogram_build_result_t* build,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t d_levels,
  uint64_t num_rows,
  uint64_t row_stride_samples,
  bool is_evenly_segmented)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_histogram_build(
      build,
      num_channels,
      num_active_channels,
      d_samples,
      num_output_levels_val,
      d_output_histograms,
      d_levels,
      num_rows,
      row_stride_samples,
      is_evenly_segmented,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));
}

void histogram_even(
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  int num_output_levels_val,
  cccl_value_t lower_level,
  cccl_value_t upper_level,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples)
{
  cccl_device_histogram_build_result_t build;
  build_histogram(
    &build, d_samples, num_output_levels_val, d_output_histograms, lower_level, num_rows, row_stride_samples, true);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_histogram_even(
      build,
      nullptr,
      &temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_histogram_even(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      0));

  REQUIRE(CUDA_SUCCESS == cccl_device_histogram_cleanup(&build));
}

// Copied from catch2_test_device_histogram.cu (With some modifications)
template <size_t ActiveChannels>
auto generate_level_counts_to_test(int max_level_count) -> std::vector<int>
{
  // first channel tests maximum number of levels, later channels less and less
  std::vector<int> r{max_level_count};
  for (size_t c = 1; c < ActiveChannels; ++c)
  {
    r[c] = r[c - 1] / 2 + 1;
  }
  return r;
}

template <size_t ActiveChannels, typename LevelT>
auto setup_bin_levels_for_even(const std::vector<int>& num_levels, LevelT max_level, int max_level_count)
  -> std::vector<std::vector<LevelT>>
{
  std::vector<std::vector<LevelT>> levels(2);
  auto& lower_level = levels[0];
  auto& upper_level = levels[1];

  lower_level.resize(ActiveChannels);
  upper_level.resize(ActiveChannels);

  // Create upper and lower levels between between [0:max_level], getting narrower with each channel. Example:
  //    max_level = 256
  //   num_levels = { 257, 129,  65 }
  //  lower_level = {   0,  64,  96 }
  //  upper_level = { 256, 192, 160 }

  const auto min_bin_width = max_level / (max_level_count - 1);
  REQUIRE(min_bin_width > 0);

  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    const int num_bins        = num_levels[c] - 1;
    const auto min_hist_width = num_bins * min_bin_width;
    lower_level[c]            = static_cast<LevelT>(max_level / 2 - min_hist_width / 2);
    upper_level[c]            = static_cast<LevelT>(max_level / 2 + min_hist_width / 2);
    REQUIRE(lower_level[c] < upper_level[c]);
  }
  return levels;
}

template <int Channels, typename counter_t, size_t ActiveChannels, typename SampleT, typename TransformOp, typename OffsetT>
auto compute_reference_result(
  const std::vector<SampleT>& h_samples,
  const TransformOp& sample_to_bin_index,
  const std::vector<int>& num_levels,
  OffsetT width,
  OffsetT height,
  OffsetT row_pitch) -> std::array<std::vector<counter_t>, ActiveChannels>
{
  auto h_histogram = std::array<std::vector<counter_t>, ActiveChannels>{};
  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    h_histogram[c].resize(num_levels[c] - 1);
  }
  for (OffsetT row = 0; row < height; ++row)
  {
    for (OffsetT pixel = 0; pixel < width; ++pixel)
    {
      for (size_t c = 0; c < ActiveChannels; ++c)
      {
        const auto offset = row * (row_pitch / sizeof(SampleT)) + pixel * Channels + c;
        const int bin     = sample_to_bin_index(static_cast<int>(c), h_samples[offset]);
        if (bin >= 0 && bin < static_cast<int>(h_histogram[c].size())) // if bin is valid
        {
          ++h_histogram[c][bin];
        }
      }
    }
  }
  return h_histogram;
}

C2H_TEST("DeviceHistogram::HistogramEven API usage", "[histogram][device]")
{
  using counter_t = int;

  int num_samples = 10;
  std::vector<float> d_samples{2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5};

  int num_rows = 1;

  int num_levels = 7;
  std::vector<int> d_num_levels{num_levels};
  std::vector<counter_t> d_single_histogram(6, 0);
  pointer_t<counter_t> d_single_histogram_ptr(d_single_histogram);

  LevelT lower_level = 0.0;
  LevelT upper_level = 12.0;

  pointer_t<float> d_samples_ptr(d_samples);
  value_t<int> num_levels_val{num_levels};
  pointer_t<int> d_num_levels_ptr(d_num_levels);

  value_t<LevelT> lower_level_val{lower_level};
  value_t<LevelT> upper_level_val{upper_level};

  size_t row_stride_samples = num_samples;

  histogram_even(
    d_samples_ptr,
    d_single_histogram_ptr,
    num_levels_val,
    num_levels,
    lower_level_val,
    upper_level_val,
    num_samples,
    num_rows,
    row_stride_samples);

  std::vector<counter_t> d_histogram_out(d_single_histogram_ptr);
  CHECK(d_histogram_out == std::vector{1, 5, 0, 3, 0, 0});
}

struct bit_and_anything
{
  template <typename T>
  _CCCL_HOST_DEVICE auto operator()(const T& a, const T& b) const -> T
  {
    using U = typename cub::Traits<T>::UnsignedBits;
    return ::cuda::std::bit_cast<T>(static_cast<U>(::cuda::std::bit_cast<U>(a) & ::cuda::std::bit_cast<U>(b)));
  }
};

C2H_TEST("DeviceHistogram::HistogramEven basic use", "[histogram][device]", sample_types)
{
  using counter_t = int;
  using sample_t  = c2h::get<0, TestType>;
  using offset_t  = int;

  const auto max_level       = LevelT{sizeof(sample_t) == 1 ? 126 : 1024};
  const auto max_level_count = (sizeof(sample_t) == 1 ? 126 : 1024) + 1;

  offset_t width  = 1920;
  offset_t height = 1080;

  constexpr int channels        = 1;
  constexpr int active_channels = 1;

  const auto padding_bytes     = static_cast<offset_t>(GENERATE(size_t{0}, 13 * sizeof(sample_t)));
  const offset_t row_pitch     = width * channels * sizeof(sample_t) + padding_bytes;
  const auto num_levels        = generate_level_counts_to_test<active_channels>(max_level_count);
  const offset_t total_samples = height * (row_pitch / sizeof(sample_t));

  std::vector<int64_t> samples_gen = generate<int64_t>(total_samples);
  std::vector<sample_t> h_samples(total_samples);
  for (int i = 0; i < total_samples; i++)
  {
    h_samples[i] = static_cast<sample_t>(samples_gen[i]);
  }

  std::vector<counter_t> d_single_histogram(num_levels[0] - 1, 0);

  auto levels = setup_bin_levels_for_even<active_channels, LevelT>(num_levels, max_level, max_level_count);

  auto& lower_level = levels[0];
  auto& upper_level = levels[1];

  // Compute reference result
  auto fp_scales = ::cuda::std::array<LevelT, active_channels>{}; // only used when LevelT is floating point
  for (size_t c = 0; c < active_channels; ++c)
  {
    if constexpr (!std::is_integral<LevelT>::value)
    {
      fp_scales[c] = static_cast<LevelT>(num_levels[c] - 1) / static_cast<LevelT>(upper_level[c] - lower_level[c]);
    }
  }

  auto sample_to_bin_index = [&](int channel, sample_t sample) {
    using common_t             = ::cuda::std::common_type_t<LevelT, sample_t>;
    const auto n               = num_levels[channel];
    const auto max             = static_cast<common_t>(upper_level[channel]);
    const auto min             = static_cast<common_t>(lower_level[channel]);
    const auto promoted_sample = static_cast<common_t>(sample);
    if (promoted_sample < min || promoted_sample >= max)
    {
      return n; // out of range
    }
    if constexpr (::cuda::std::is_integral<LevelT>::value)
    {
      // Accurate bin computation following the arithmetic we guarantee in the HistoEven docs
      return static_cast<int>(
        static_cast<uint64_t>(promoted_sample - min) * static_cast<uint64_t>(n - 1) / static_cast<uint64_t>(max - min));
    }
    else
    {
      return static_cast<int>((static_cast<common_t>(sample) - min) * fp_scales[channel]);
    }
    _CCCL_UNREACHABLE();
  };
  auto h_histogram = compute_reference_result<channels, counter_t, active_channels>(
    h_samples, sample_to_bin_index, num_levels, width, height, row_pitch);

  // Compute result and verify
  pointer_t<sample_t> sample_ptr(h_samples);
  pointer_t<counter_t> d_single_histogram_ptr(d_single_histogram);

  value_t<int> num_levels_val{num_levels[0]};
  value_t<LevelT> lower_level_val{lower_level[0]};
  value_t<LevelT> upper_level_val{upper_level[0]};

  histogram_even(
    sample_ptr,
    d_single_histogram_ptr,
    num_levels_val,
    num_levels[0],
    lower_level_val,
    upper_level_val,
    width,
    height,
    row_pitch / sizeof(sample_t));

  for (size_t c = 0; c < active_channels; ++c)
  {
    CHECK(h_histogram[c] == std::vector<counter_t>(d_single_histogram_ptr));
  }
}

C2H_TEST("DeviceHistogram::HistogramEven sample iterator", "[histogram][device]")
{
  using counter_t = int;
  using sample_t  = std::int32_t;
  using offset_t  = int;

  const auto max_level_count = 1025;

  const auto num_levels = generate_level_counts_to_test<num_active_channels>(max_level_count);
  const int num_bins    = num_levels[0] - 1;

  const offset_t samples_per_bin        = 10;
  const offset_t adjusted_total_samples = num_bins * samples_per_bin;

  // Set up iterator that counts from 0 to adjusted_total_samples - 1
  iterator_t<sample_t, counting_iterator_state_t<sample_t>> counting_it = make_counting_iterator<sample_t>("int");
  counting_it.state.value                                               = static_cast<sample_t>(0);

  std::vector<counter_t> d_single_histogram(num_levels[0] - 1, 0);

  // Set up levels so that values 0 to adjusted_total_samples-1 are evenly distributed
  std::vector<std::vector<LevelT>> levels(2);
  auto& lower_level = levels[0];
  auto& upper_level = levels[1];

  lower_level.resize(num_active_channels);
  upper_level.resize(num_active_channels);

  lower_level[0] = static_cast<LevelT>(0);
  upper_level[0] = static_cast<LevelT>(adjusted_total_samples);

  // Compute reference result - each bin should have exactly samples_per_bin elements
  auto h_histogram = std::array<std::vector<counter_t>, num_active_channels>{};
  h_histogram[0].resize(num_levels[0] - 1, samples_per_bin);

  // Compute result and verify
  pointer_t<counter_t> d_single_histogram_ptr(d_single_histogram);

  value_t<int> num_levels_val{num_levels[0]};
  value_t<LevelT> lower_level_val{lower_level[0]};
  value_t<LevelT> upper_level_val{upper_level[0]};

  histogram_even(
    counting_it,
    d_single_histogram_ptr,
    num_levels_val,
    num_levels[0],
    lower_level_val,
    upper_level_val,
    adjusted_total_samples,
    1,
    adjusted_total_samples);

  for (size_t c = 0; c < num_active_channels; ++c)
  {
    CHECK(h_histogram[c] == std::vector<counter_t>(d_single_histogram_ptr));
  }
}
