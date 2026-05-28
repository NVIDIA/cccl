// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/host_vector.h>

#include <random>

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

  // Skip invalid configurations; see range.cu for rationale.
  if (num_bins > max_representable_bins<SampleT>())
  {
    state.skip("Number of bins exceeds what SampleT can represent");
    return;
  }

  const SampleT lower_level = get_lower_level<SampleT>();
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  // Jittered uniform spacing keeps DispatchRange on the SearchTransform path
  // while keeping bin widths within ~2x of each other. Fixed seed makes the
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
      lvl = static_cast<SampleT>(h_levels[i - 1] + SampleT{1});
    }
    h_levels[i] = lvl;
  }
  if (h_levels[num_bins] <= h_levels[num_bins - 1])
  {
    h_levels[num_bins] = static_cast<SampleT>(h_levels[num_bins - 1] + SampleT{1});
  }
  thrust::device_vector<SampleT> levels_r = h_levels;
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

  // Warmup + correctness check: run MultiHistogramRange once outside
  // `state.exec`, checking the dispatch return code, then verify each
  // channel's histogram bin-by-bin against an independent reference.
  // Skipped when CUB_BENCH_HISTOGRAM_VERIFY=0|false|no|off.
  if (bench_correctness_checks_enabled())
  {
    thrust::fill(hist_r.begin(), hist_r.end(), CounterT{0});
    thrust::fill(hist_g.begin(), hist_g.end(), CounterT{0});
    thrust::fill(hist_b.begin(), hist_b.end(), CounterT{0});
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    bench_check_cuda(
      (cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        cuda::std::array<CounterT*, num_active_channels>{d_histogram_r, d_histogram_g, d_histogram_b},
        cuda::std::array<int, num_active_channels>{num_levels_r, num_levels_g, num_levels_b},
        cuda::std::array<const SampleT*, num_active_channels>{d_levels_r, d_levels_g, d_levels_b},
        static_cast<OffsetT>(elements))),
      "warmup MultiHistogramRange temp-size");
    thrust::device_vector<unsigned char> warmup_tmp(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(warmup_tmp.data());
    bench_check_cuda(
      (cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        cuda::std::array<CounterT*, num_active_channels>{d_histogram_r, d_histogram_g, d_histogram_b},
        cuda::std::array<int, num_active_channels>{num_levels_r, num_levels_g, num_levels_b},
        cuda::std::array<const SampleT*, num_active_channels>{d_levels_r, d_levels_g, d_levels_b},
        static_cast<OffsetT>(elements))),
      "warmup MultiHistogramRange");
    bench_check_cuda(cudaDeviceSynchronize(), "warmup sync");

    std::vector<thrust::device_vector<CounterT>> opt_hists_d;
    opt_hists_d.emplace_back(std::move(hist_r));
    opt_hists_d.emplace_back(std::move(hist_g));
    opt_hists_d.emplace_back(std::move(hist_b));
    std::vector<thrust::device_vector<SampleT>> d_levels_per_channel;
    d_levels_per_channel.emplace_back(std::move(levels_r));
    d_levels_per_channel.emplace_back(std::move(levels_g));
    d_levels_per_channel.emplace_back(std::move(levels_b));
    bench_verify_histogram_range<num_channels, num_active_channels, SampleT, CounterT, OffsetT>(
      input, opt_hists_d, d_levels_per_channel, static_cast<OffsetT>(elements), "multi.range");
    hist_r        = std::move(opt_hists_d[0]);
    hist_g        = std::move(opt_hists_d[1]);
    hist_b        = std::move(opt_hists_d[2]);
    d_histogram_r = thrust::raw_pointer_cast(hist_r.data());
    d_histogram_g = thrust::raw_pointer_cast(hist_g.data());
    d_histogram_b = thrust::raw_pointer_cast(hist_b.data());
    levels_r      = std::move(d_levels_per_channel[0]);
    levels_g      = std::move(d_levels_per_channel[1]);
    levels_b      = std::move(d_levels_per_channel[2]);
    d_levels_r    = thrust::raw_pointer_cast(levels_r.data());
    d_levels_g    = thrust::raw_pointer_cast(levels_g.data());
    d_levels_b    = thrust::raw_pointer_cast(levels_b.data());
  }

  caching_allocator_t alloc;
  // Force the persisting-L2 reservation back to 0 and demote any persisting
  // lines outside the timed window, so neither cudaAccessPolicyWindow nor a
  // bumped cudaLimitPersistingL2CacheSize can carry across iterations. The
  // default reservation is 0; hardcoding 0 also clears any pollution left by
  // a prior benchmark in the same nvbench process.
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
               cudaCtxResetPersistingL2Cache();
               timer.start();
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
               timer.stop();
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
  .add_int64_axis("Elements{io}", {100'000, 1 << 20, 20'000'000, 1 << 28})
  .add_int64_axis("Bins", {32, 100, 2000, 16384, 60000, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"});
