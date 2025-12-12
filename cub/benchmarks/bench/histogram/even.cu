// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

#include "histogram_common.cuh"

// %RANGE% TUNE_ITEMS ipt 4:28:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_LOAD_ALGORITHM_ID laid 0:2:1
// %RANGE% TUNE_VEC_SIZE_POW vec 0:2:1

template <typename SampleT, typename CounterT, typename OffsetT>
static void even(nvbench::state& state, nvbench::type_list<SampleT, CounterT, OffsetT>)
{
  constexpr int num_channels        = 1;
  constexpr int num_active_channels = 1;

  using sample_iterator_t = SampleT*;

#if !TUNE_BASE
  using policy_t = policy_hub_t<key_t, num_channels, num_active_channels>;
  using dispatch_t =
    cub::DispatchHistogram<num_channels, //
                           num_active_channels,
                           sample_iterator_t,
                           CounterT,
                           SampleT,
                           OffsetT,
                           policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchHistogram<num_channels, //
                           num_active_channels,
                           sample_iterator_t,
                           CounterT,
                           /* LevelT = */ SampleT,
                           OffsetT>;
#endif // TUNE_BASE

  const auto entropy   = str_to_entropy(state.get_string("Entropy"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;

  // Skip invalid configurations where LevelT (= SampleT) cannot represent the number of bins
  if constexpr (cuda::std::is_integral_v<SampleT>)
  {
    if (num_bins > static_cast<int64_t>(cuda::std::numeric_limits<SampleT>::max()))
    {
      state.skip("Number of bins exceeds what LevelT (= SampleT) can represent");
      return;
    }
  }

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  thrust::device_vector<SampleT> input = generate(elements, entropy, lower_level, upper_level);
  thrust::device_vector<CounterT> hist(num_bins);

  SampleT* d_input      = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  std::uint8_t* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};

  cuda::std::bool_constant<sizeof(SampleT) == 1> is_byte_sample;
  OffsetT num_row_pixels     = static_cast<OffsetT>(elements);
  OffsetT num_rows           = 1;
  OffsetT row_stride_samples = num_row_pixels;

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

  dispatch_t::DispatchEven(
    d_temp_storage,
    temp_storage_bytes,
    d_input,
    {d_histogram},
    {num_levels},
    {lower_level},
    {upper_level},
    num_row_pixels,
    num_rows,
    row_stride_samples,
    0,
    is_byte_sample);

  thrust::device_vector<nvbench::uint8_t> tmp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::DispatchEven(
      d_temp_storage,
      temp_storage_bytes,
      d_input,
      {d_histogram},
      {num_levels},
      {lower_level},
      {upper_level},
      num_row_pixels,
      num_rows,
      row_stride_samples,
      launch.get_stream(),
      is_byte_sample);
  });
}

using counter_types     = nvbench::type_list<int32_t>;
using some_offset_types = nvbench::type_list<int32_t>;

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(even, NVBENCH_TYPE_AXES(sample_types, counter_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Bins", {32, 128, 2048, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"});
