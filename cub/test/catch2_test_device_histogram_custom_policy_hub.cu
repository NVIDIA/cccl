// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_histogram.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/array>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the histogram dispatcher after publishing the tuning API

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct my_policy_hub
{
  // simplified from Policy500 of the CUB histogram tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using AgentHistogramPolicyT = AgentHistogramPolicy<384, 16, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };
};

C2H_TEST("DispatchHistogram::DispatchEven: custom policy hub", "[histogram][device]")
{
  using sample_t                    = cuda::std::uint8_t;
  using counter_t                   = int;
  using level_t                     = int;
  using offset_t                    = int;
  constexpr int num_channels        = 1;
  constexpr int num_active_channels = 1;
  constexpr int num_bins            = 16;
  const offset_t num_row_pixels     = 256;
  const offset_t num_rows           = 1;
  const offset_t row_stride_samples = num_row_pixels * num_channels;
  const int num_output_levels       = num_bins + 1;

  c2h::host_vector<sample_t> h_samples(num_row_pixels);
  for (offset_t i = 0; i < num_row_pixels; ++i)
  {
    h_samples[i] = static_cast<sample_t>(i % num_bins);
  }

  c2h::device_vector<sample_t> d_samples = h_samples;
  c2h::device_vector<counter_t> d_histogram(num_bins, 0);

  cuda::std::array<counter_t*, num_active_channels> d_histograms{thrust::raw_pointer_cast(d_histogram.data())};
  cuda::std::array<int, num_active_channels> num_levels{num_output_levels};
  cuda::std::array<level_t, num_active_channels> lower_level{0};
  cuda::std::array<level_t, num_active_channels> upper_level{num_bins};

  using policy_hub_t = my_policy_hub<sample_t, counter_t, num_channels, num_active_channels, /* is_even */ true>;
  using dispatch_t =
    DispatchHistogram<num_channels, num_active_channels, sample_t*, counter_t, level_t, offset_t, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::DispatchEven(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histograms,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_samples,
    /* stream */ nullptr,
    cuda::std::true_type{});
  c2h::device_vector<std::uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::DispatchEven(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_samples.data()),
    d_histograms,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_samples,
    /* stream */ nullptr,
    cuda::std::true_type{});

  c2h::host_vector<counter_t> expected_histogram(num_bins, 0);
  for (const auto sample : h_samples)
  {
    ++expected_histogram[sample];
  }

  c2h::host_vector<counter_t> h_histogram = d_histogram;
  REQUIRE(h_histogram == expected_histogram);
}
