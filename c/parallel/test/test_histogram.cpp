//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>

#include "test_util.h"
#include <cccl/c/histogram.h>

using sample_types = c2h::type_list<std::int8_t, std::uint16_t, std::int32_t, std::uint64_t, float, double>;
using LevelT       = double;

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

  const std::string sass = inspect_sass(build->cubin, build->cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);
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

void histogram_range(
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  int num_output_levels_val,
  cccl_value_t d_levels,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples)
{
  cccl_device_histogram_build_result_t build;
  build_histogram(
    &build, d_samples, num_output_levels_val, d_output_histograms, d_levels, num_rows, row_stride_samples, true);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_histogram_range(
      build,
      nullptr,
      &temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_histogram_range(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      0));

  REQUIRE(CUDA_SUCCESS == cccl_device_histogram_cleanup(&build));
}

// Copied from catch2_test_device_histogram.cu (With some modifications)
template <typename T>
auto cast_if_half_pointer(T* p) -> T*
{
  return p;
}

template <size_t ActiveChannels>
auto generate_level_counts_to_test(int max_level_count) -> std::vector<int>
{
  // TODO(bgruber): eventually, just pick a random number of levels per channel

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

  // TODO(bgruber): eventually, we could just pick a random lower/upper bound for each channel

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

template <int Channels, typename CounterT, size_t ActiveChannels, typename SampleT, typename TransformOp, typename OffsetT>
auto compute_reference_result(
  const std::vector<SampleT>& h_samples,
  const TransformOp& sample_to_bin_index,
  const std::vector<int>& num_levels,
  OffsetT width,
  OffsetT height,
  OffsetT row_pitch) -> std::array<std::vector<CounterT>, ActiveChannels>
{
  auto h_histogram = std::array<std::vector<CounterT>, ActiveChannels>{};
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
        // TODO(bgruber): use an mdspan to access h_samples
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

template <typename T, size_t N>
auto to_array_of_ptrs(std::array<c2h::device_vector<T>, N>& in)
{
  std::array<decltype(cast_if_half_pointer(std::declval<T*>())), N> r;
  for (size_t i = 0; i < N; i++)
  {
    r[i] = cast_if_half_pointer(thrust::raw_pointer_cast(in[i].data()));
  }
  return r;
}

template <size_t ActiveChannels, typename LevelT>
auto setup_bin_levels_for_range(const std::vector<int>& num_levels, LevelT max_level, int max_level_count)
  -> std::vector<std::vector<LevelT>>
{
  // TODO(bgruber): eventually, we could just pick random levels for each channel

  const auto min_bin_width = max_level / (max_level_count - 1);
  REQUIRE(min_bin_width > 0);

  std::vector<std::vector<LevelT>> levels(ActiveChannels);
  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    levels[c].resize(num_levels[c]);
    const int num_bins        = num_levels[c] - 1;
    const auto min_hist_width = num_bins * min_bin_width;
    const auto lower_level    = (max_level / 2 - min_hist_width / 2);
    for (int l = 0; l < num_levels[c]; ++l)
    {
      levels[c][l] = static_cast<LevelT>(lower_level + l * min_bin_width);
      if (l > 0)
      {
        REQUIRE(levels[c][l - 1] < levels[c][l]);
      }
    }
  }
  return levels;
}

C2H_TEST("DeviceHistogram::HistogramEven API usage", "[histogram][device]")
{
  using CounterT = int;

  int num_samples = 10;
  std::vector<float> d_samples{2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5};

  int num_rows = 1;

  int num_levels = 7;
  std::vector<int> d_num_levels{num_levels};
  std::vector<CounterT> d_single_histogram(6, 0);
  pointer_t<CounterT> d_single_histogram_ptr(d_single_histogram);

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

  std::vector<CounterT> d_histogram_out(d_single_histogram_ptr);
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
  using CounterT = int;
  using sample_t = c2h::get<0, TestType>;
  using offset_t = int;

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
  // c2h::device_vector<sample_t> d_samples;
  // d_samples.resize(total_samples);
  // int entropy_reduction = 0;

  // if (entropy_reduction >= 0)
  // {
  //   c2h::gen(C2H_SEED(1), d_samples, sample_t{0}, static_cast<sample_t>(max_level));
  //   if (entropy_reduction > 0)
  //   {
  //     c2h::device_vector<sample_t> tmp(d_samples.size());
  //     for (int i = 0; i < entropy_reduction; ++i)
  //     {
  //       c2h::gen(C2H_SEED(1), tmp);
  //       thrust::transform(
  //         c2h::device_policy, d_samples.cbegin(), d_samples.cend(), tmp.cbegin(), d_samples.begin(),
  //         bit_and_anything{});
  //     }
  //   }
  // }

  // auto h_samples = c2h::host_vector<sample_t>(d_samples);

  //   ::cuda::std::array<c2h::device_vector<CounterT>, active_channels> d_histogram;

  //   for (size_t c = 0; c < active_channels; ++c)
  //   {
  //     d_histogram[c].resize(num_levels[c] - 1);
  //   }

  std::vector<CounterT> d_single_histogram(num_levels[0] - 1, 0);

  auto levels = setup_bin_levels_for_even<active_channels, LevelT>(num_levels, max_level, max_level_count);

  auto& lower_level = levels[0];
  auto& upper_level = levels[1];

  // Compute reference result
  auto fp_scales = ::cuda::std::array<LevelT, active_channels>{}; // only used when LevelT is floating point
  std::ignore    = fp_scales; // casting to void was insufficient. TODO(bgruber): use [[maybe_unsued]] in C++17
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
      return static_cast<int>((sample - min) * fp_scales[channel]);
    }
    _CCCL_UNREACHABLE();
  };
  auto h_histogram = compute_reference_result<channels, CounterT, active_channels>(
    h_samples, sample_to_bin_index, num_levels, width, height, row_pitch);

  // Compute result and verify
  {
    pointer_t<sample_t> sample_ptr(h_samples);
    pointer_t<CounterT> d_single_histogram_ptr(d_single_histogram);

    value_t<int> num_levels_val{num_levels[0]};
    value_t<LevelT> lower_level_val{lower_level[0]};
    value_t<LevelT> upper_level_val{upper_level[0]};

    if constexpr (active_channels == 1 && channels == 1)
    {
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
    }
    else
    {
      //   // new API entry-point
      //   multi_histogram_even<channels, active_channels>(
      //     sample_ptr,
      //     to_array_of_ptrs(d_histogram),
      //     num_levels,
      //     cast_if_half(lower_level),
      //     cast_if_half(upper_level),
      //     width,
      //     height,
      //     row_pitch);
    }
    for (size_t c = 0; c < active_channels; ++c)
    {
      CHECK(h_histogram[c] == std::vector<CounterT>(d_single_histogram_ptr));
    }
  }
}

// C2H_TEST("DeviceHistogram::HistogramRange basic use", "[histogram][device]", sample_types)
// {
//   using CounterT = int;
//   using sample_t = c2h::get<0, TestType>;
//   using offset_t = int;

//   const auto max_level       = LevelT{sizeof(sample_t) == 1 ? 126 : 1024};
//   const auto max_level_count = (sizeof(sample_t) == 1 ? 126 : 1024) + 1;

//   offset_t width  = 1920;
//   offset_t height = 1080;

//   constexpr int channels        = 1;
//   constexpr int active_channels = 1;

//   // const auto padding_bytes     = static_cast<offset_t>(GENERATE(size_t{0}, 13 * sizeof(sample_t)));
//   const auto padding_bytes     = static_cast<offset_t>(size_t{0});
//   const offset_t row_pitch     = width * channels * sizeof(sample_t) + padding_bytes;
//   const auto num_levels        = generate_level_counts_to_test<active_channels>(max_level_count);
//   const offset_t total_samples = height * (row_pitch / sizeof(sample_t));

//   std::vector<int64_t> samples_gen = generate<int64_t>(total_samples);
//   std::vector<sample_t> h_samples(total_samples);
//   for (int i = 0; i < total_samples; i++)
//   {
//     h_samples[i] = static_cast<sample_t>(samples_gen[i]);
//   }
//   // c2h::device_vector<sample_t> d_samples;
//   // d_samples.resize(total_samples);
//   // int entropy_reduction = 0;

//   // if (entropy_reduction >= 0)
//   // {
//   //   c2h::gen(C2H_SEED(1), d_samples, sample_t{0}, static_cast<sample_t>(max_level));
//   //   if (entropy_reduction > 0)
//   //   {
//   //     c2h::device_vector<sample_t> tmp(d_samples.size());
//   //     for (int i = 0; i < entropy_reduction; ++i)
//   //     {
//   //       c2h::gen(C2H_SEED(1), tmp);
//   //       thrust::transform(
//   //         c2h::device_policy, d_samples.cbegin(), d_samples.cend(), tmp.cbegin(), d_samples.begin(),
//   //         bit_and_anything{});
//   //     }
//   //   }
//   // }

//   // auto h_samples = c2h::host_vector<sample_t>(d_samples);

//   //   ::cuda::std::array<c2h::device_vector<CounterT>, active_channels> d_histogram;

//   //   for (size_t c = 0; c < active_channels; ++c)
//   //   {
//   //     d_histogram[c].resize(num_levels[c] - 1);
//   //   }

//   std::vector<CounterT> d_single_histogram(num_levels[0] - 1, 0);

//   auto h_levels = setup_bin_levels_for_range<active_channels, LevelT>(num_levels, max_level, max_level_count);

//   // Compute reference result
//   const auto sample_to_bin_index = [&](int channel, sample_t sample) {
//     const auto* l  = h_levels[channel].data();
//     const auto n   = static_cast<int>(h_levels[channel].size());
//     const auto* ub = std::upper_bound(l, l + n, static_cast<LevelT>(sample));
//     return ub == l /* sample smaller than first bin */ ? n : static_cast<int>(std::distance(l, ub) - 1);
//   };
//   auto h_histogram = compute_reference_result<channels, CounterT, active_channels>(
//     h_samples, sample_to_bin_index, num_levels, width, height, row_pitch);

//   // Compute result and verify
//   {
//     pointer_t<sample_t> sample_ptr(h_samples);
//     pointer_t<CounterT> d_single_histogram_ptr(d_single_histogram);

//     pointer_t<int> num_levels_ptr(num_levels);

//     // cccl_iterator_t levels_ptr{
//     //   sizeof(LevelT),
//     //   alignof(LevelT),
//     //   cccl_iterator_kind_t::CCCL_POINTER,
//     //   {},
//     //   {},
//     //   get_type_info<LevelT>(),
//     //   h_levels[0].data()};
//     pointer_t<LevelT> levels_ptr(h_levels[0]);

//     if constexpr (active_channels == 1 && channels == 1)
//     {
//       histogram_range(
//         sample_ptr,
//         d_single_histogram_ptr,
//         get_type_info<CounterT>(),
//         num_levels_ptr,
//         num_levels[0],
//         levels_ptr,
//         width,
//         height,
//         row_pitch / sizeof(sample_t));
//     }
//     else
//     {
//       //   // new API entry-point
//       //   multi_histogram_even<channels, active_channels>(
//       //     sample_ptr,
//       //     to_array_of_ptrs(d_histogram),
//       //     num_levels,
//       //     cast_if_half(lower_level),
//       //     cast_if_half(upper_level),
//       //     width,
//       //     height,
//       //     row_pitch);
//     }
//     for (size_t c = 0; c < active_channels; ++c)
//     {
//       CHECK(h_histogram[c] == std::vector<CounterT>(d_single_histogram_ptr));
//     }
//   }
// }
