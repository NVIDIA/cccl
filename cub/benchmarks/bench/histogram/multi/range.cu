// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/sequence.h>

#include <nvbench_helper.cuh>

#include "../histogram_common.cuh"

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_LOAD_ALGORITHM_ID laid 0:2:1
// %RANGE% TUNE_VEC_SIZE_POW vec 0:2:1

template <typename SampleT, typename CounterT, typename OffsetT>
static void range(nvbench::state& state, nvbench::type_list<SampleT, CounterT, OffsetT>)
{
  constexpr int num_channels        = 4;
  constexpr int num_active_channels = 3;

  const auto entropy     = str_to_entropy(state.get_string("Entropy"));
  const auto elements    = state.get_int64("Elements{io}");
  const auto num_bins    = state.get_int64("Bins");
  const int num_levels_r = static_cast<int>(num_bins) + 1;
  const int num_levels_g = num_levels_r;
  const int num_levels_b = num_levels_g;

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  SampleT step = (upper_level - lower_level) / num_bins;
  thrust::device_vector<SampleT> levels_r(num_bins + 1);

  // TODO Extract sequence to the helper TU
  thrust::sequence(levels_r.begin(), levels_r.end(), lower_level, step);
  thrust::device_vector<SampleT> levels_g = levels_r;
  thrust::device_vector<SampleT> levels_b = levels_g;

  SampleT* d_levels_r = thrust::raw_pointer_cast(levels_r.data());
  SampleT* d_levels_g = thrust::raw_pointer_cast(levels_g.data());
  SampleT* d_levels_b = thrust::raw_pointer_cast(levels_b.data());

  thrust::device_vector<CounterT> hist_r(num_bins);
  thrust::device_vector<CounterT> hist_g(num_bins);
  thrust::device_vector<CounterT> hist_b(num_bins);
  thrust::device_vector<SampleT> input = generate(elements * num_channels, entropy, lower_level, upper_level);

  SampleT* d_input        = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram_r = thrust::raw_pointer_cast(hist_r.data());
  CounterT* d_histogram_g = thrust::raw_pointer_cast(hist_g.data());
  CounterT* d_histogram_b = thrust::raw_pointer_cast(hist_b.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements * num_active_channels);
  state.add_global_memory_writes<CounterT>(num_bins * num_active_channels);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_policy_selector<key_t, num_channels, num_active_channels>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      (cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>),
      "MultiHistogramRange failed",
      d_input,
      cuda::std::array<CounterT*, num_active_channels>{d_histogram_r, d_histogram_g, d_histogram_b},
      cuda::std::array<int, num_active_channels>{num_levels_r, num_levels_g, num_levels_b},
      cuda::std::array<const SampleT*, num_active_channels>{d_levels_r, d_levels_g, d_levels_b},
      static_cast<OffsetT>(elements),
      env);
  });
}

using counter_types     = nvbench::type_list<int32_t>;
using some_offset_types = nvbench::type_list<int32_t>;

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(range, NVBENCH_TYPE_AXES(sample_types, counter_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Bins", {32, 128, 2048, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"});
