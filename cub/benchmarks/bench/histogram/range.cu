// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/host_vector.h>

#include <random>

#include <nvbench_helper.cuh>

#include "histogram_common.cuh"
#include "histogram_inputs.cuh"

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
  const auto shape     = parse_input_shape(state.get_string("InputShape"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;

  if (elements > static_cast<int64_t>(::cuda::std::numeric_limits<OffsetT>::max()))
  {
    state.skip("Number of elements overflows OffsetT");
    return;
  }

  // Skip configurations whose bin count cannot fit in SampleT's value range.
  if (num_bins > max_representable_bins<SampleT>())
  {
    state.skip("Number of bins exceeds what SampleT can represent");
    return;
  }

  const SampleT lower_level = get_lower_level<SampleT>();
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  // Jittered uniform spacing keeps DispatchRange on the SearchTransform path
  // while keeping bin widths within ~3x of each other. Fixed seed makes the
  // levels reproducible across runs.
  thrust::host_vector<SampleT> h_levels(num_bins + 1);
  const double L    = static_cast<double>(lower_level);
  const double U    = static_cast<double>(upper_level);
  const double step = (U - L) / static_cast<double>(num_bins);
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<double> jitter(-0.25, 0.25);
  h_levels[0]        = lower_level;
  h_levels[num_bins] = upper_level;
  for (int i = 1; i < num_bins; ++i)
  {
    SampleT lvl = static_cast<SampleT>(L + i * step + step * jitter(rng));
    if (lvl <= h_levels[i - 1])
    {
      if constexpr (::cuda::std::is_integral_v<SampleT>)
      {
        if (h_levels[i - 1] == ::cuda::std::numeric_limits<SampleT>::max())
        {
          state.skip("SampleT range exhausted while constructing levels");
          return;
        }
      }
      lvl = static_cast<SampleT>(h_levels[i - 1] + SampleT{1});
    }
    h_levels[i] = lvl;
  }
  if (h_levels[num_bins] <= h_levels[num_bins - 1])
  {
    if constexpr (::cuda::std::is_integral_v<SampleT>)
    {
      if (h_levels[num_bins - 1] == ::cuda::std::numeric_limits<SampleT>::max())
      {
        state.skip("SampleT range exhausted while constructing levels");
        return;
      }
    }
    h_levels[num_bins] = static_cast<SampleT>(h_levels[num_bins - 1] + SampleT{1});
  }
  thrust::device_vector<SampleT> levels = h_levels;
  SampleT* d_levels                     = thrust::raw_pointer_cast(levels.data());

  thrust::device_vector<SampleT> input =
    generate_histogram_input_range<SampleT>(shape, elements, static_cast<int>(num_bins), d_levels);
  thrust::device_vector<CounterT> hist(num_bins);

  SampleT* d_input      = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

  // Optionally validate one untimed invocation.
  if (bench_correctness_checks_enabled())
  {
    thrust::fill(hist.begin(), hist.end(), CounterT{0});
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    bench_check_cuda(
      cub::DeviceHistogram::HistogramRange(
        d_temp_storage, temp_storage_bytes, d_input, d_histogram, num_levels, d_levels, static_cast<OffsetT>(elements)),
      "warmup HistogramRange temp-size");
    thrust::device_vector<unsigned char> warmup_tmp(std::max(temp_storage_bytes, size_t{1}));
    d_temp_storage = thrust::raw_pointer_cast(warmup_tmp.data());
    bench_check_cuda(
      cub::DeviceHistogram::HistogramRange(
        d_temp_storage, temp_storage_bytes, d_input, d_histogram, num_levels, d_levels, static_cast<OffsetT>(elements)),
      "warmup HistogramRange");
    bench_check_cuda(cudaDeviceSynchronize(), "warmup sync");

    std::vector<thrust::device_vector<CounterT>> opt_hists_d;
    opt_hists_d.emplace_back(std::move(hist));
    std::vector<thrust::device_vector<SampleT>> d_levels_per_channel;
    d_levels_per_channel.emplace_back(std::move(levels));
    bench_verify_histogram_range<1, 1, SampleT, CounterT, OffsetT>(
      input, opt_hists_d, d_levels_per_channel, static_cast<OffsetT>(elements), "range");
    hist        = std::move(opt_hists_d[0]);
    d_histogram = thrust::raw_pointer_cast(hist.data());
    levels      = std::move(d_levels_per_channel[0]);
    d_levels    = thrust::raw_pointer_cast(levels.data());
  }

  caching_allocator_t alloc;

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_policy_selector<key_t, 1, 1>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceHistogram::HistogramRange,
      "HistogramRange failed",
      d_input,
      d_histogram,
      num_levels,
      d_levels,
      static_cast<OffsetT>(elements),
      env);
  });
}

// Allow dedicated builds to select 64-bit counters and offsets.
#ifdef TUNE_CounterT
using counter_types = nvbench::type_list<TUNE_CounterT>;
#else // !defined(TUNE_CounterT)
using counter_types = nvbench::type_list<int32_t>;
#endif // TUNE_CounterT
#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else // !defined(TUNE_OffsetT)
using some_offset_types = nvbench::type_list<int32_t>;
#endif // TUNE_OffsetT

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(range, NVBENCH_TYPE_AXES(sample_types, counter_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {65536, 4000000, 67000000})
  .add_int64_axis("Bins", {33, 2048, 16384, 2000003})
  .add_string_axis("InputShape", {"concentrated:1.0", "concentrated:0.5", "strided_sweep"});
