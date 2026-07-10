// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/host_vector.h>

#include <random>

#include <nvbench_helper.cuh>

#include "../histogram_common.cuh"
#include "../histogram_inputs.cuh"

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

  const auto shape       = parse_input_shape(state.get_string("InputShape"));
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

  // Skip when row_stride_samples (= elements * num_channels) would overflow
  // OffsetT. See multi/even.cu for the rationale.
  if (static_cast<int64_t>(elements) * num_channels > static_cast<int64_t>(::cuda::std::numeric_limits<OffsetT>::max()))
  {
    state.skip("Row stride samples (elements * num_channels) overflows OffsetT");
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

  // Keep all channel outputs in one allocation and choose their base addresses
  // deliberately. With a power-of-two bin count, tightly packed channel chunks
  // have a power-of-two byte stride (1 MiB at 262144 int32 bins), which can send
  // each channel's spill atomics to the same subset of L2 partitions. Align each
  // completed chunk to a 128-byte cache line, then add three cache lines of
  // padding. The odd, non-power-of-two three-line phase shift breaks that stride
  // resonance while keeping every output cache-line aligned. Padding is outside
  // the ranges passed to CUB and is therefore not part of the benchmark output.
  constexpr std::size_t cache_line_bytes      = 128;
  constexpr std::size_t channel_padding_lines = 3;
  static_assert(cache_line_bytes % sizeof(CounterT) == 0);
  static_assert((channel_padding_lines & (channel_padding_lines - 1)) != 0);
  constexpr std::size_t counters_per_cache_line  = cache_line_bytes / sizeof(CounterT);
  constexpr std::size_t channel_padding_counters = channel_padding_lines * counters_per_cache_line;

  // MultiHistogramRange permits each channel to have a different number of
  // levels. Build the layout from each channel's own level count rather than
  // assuming that all chunks have the red channel's size.
  const cuda::std::array<std::size_t, num_active_channels> histogram_bins{
    static_cast<std::size_t>(num_levels_r - 1),
    static_cast<std::size_t>(num_levels_g - 1),
    static_cast<std::size_t>(num_levels_b - 1)};
  cuda::std::array<std::size_t, num_active_channels> histogram_offsets{};
  std::size_t histogram_storage_size = 0;
  for (int channel = 0; channel < num_active_channels; ++channel)
  {
    histogram_offsets[channel] = histogram_storage_size;
    histogram_storage_size += histogram_bins[channel];
    if (channel + 1 < num_active_channels)
    {
      histogram_storage_size =
        ((histogram_storage_size + counters_per_cache_line - 1) / counters_per_cache_line) * counters_per_cache_line
        + channel_padding_counters;
    }
  }
  thrust::device_vector<CounterT> histogram_storage(histogram_storage_size);
  // Per-block direct-atomic SMEM cache slot count S for the multi-channel RANGE path,
  // queried from CUB's occupancy sizer so hash_synonym, stale_resident, and poison
  // track the ACTUAL cache. Byte extent = sizeof(SampleT) *
  // num_channels * elements (the row stride the facade uses) so the int vs wide-OffsetT
  // kernel matches the dispatch at this N.
#if defined(CUB_HISTO_BENCH_DISABLE_CACHE_QUERY)
  const int64_t cache_slots = 0;
#else
  const int64_t cache_slots = cub::detail::histogram::query_direct_atomic_cache_slots_for_extent<
    num_channels,
    num_active_channels,
    /*IsEven=*/false,
    SampleT,
    CounterT,
    SampleT,
    OffsetT>(static_cast<unsigned long long>(sizeof(SampleT)) * static_cast<unsigned long long>(num_channels)
             * static_cast<unsigned long long>(elements));
#endif
  thrust::device_vector<SampleT> input = generate_histogram_input_range<SampleT>(
    shape,
    elements * num_channels,
    static_cast<int>(num_bins),
    d_levels_r,
    /*seed=*/42,
    cache_slots,
    /*sample_stride=*/num_channels);

  SampleT* d_input              = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram_storage = thrust::raw_pointer_cast(histogram_storage.data());
  const cuda::std::array<CounterT*, num_active_channels> d_histograms{
    d_histogram_storage + histogram_offsets[0],
    d_histogram_storage + histogram_offsets[1],
    d_histogram_storage + histogram_offsets[2]};

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements * num_active_channels);
  state.add_global_memory_writes<CounterT>(histogram_bins[0] + histogram_bins[1] + histogram_bins[2]);

  // Warmup + correctness check: run MultiHistogramRange once outside
  // `state.exec`, checking the dispatch return code, then verify each
  // channel's histogram sums to the channel's input sample count and matches
  // an independent reference bin-by-bin.
  // Skipped when CUB_BENCH_HISTOGRAM_VERIFY=0|false|no|off.
  if (bench_correctness_checks_enabled())
  {
    thrust::fill(histogram_storage.begin(), histogram_storage.end(), CounterT{0});
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    bench_check_cuda(
      (cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        d_histograms,
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
        d_histograms,
        cuda::std::array<int, num_active_channels>{num_levels_r, num_levels_g, num_levels_b},
        cuda::std::array<const SampleT*, num_active_channels>{d_levels_r, d_levels_g, d_levels_b},
        static_cast<OffsetT>(elements))),
      "warmup MultiHistogramRange");
    bench_check_cuda(cudaDeviceSynchronize(), "warmup sync");

    std::vector<thrust::device_vector<CounterT>> opt_hists_d;
    opt_hists_d.reserve(num_active_channels);
    for (int channel = 0; channel < num_active_channels; ++channel)
    {
      const auto first = histogram_storage.begin() + histogram_offsets[channel];
      opt_hists_d.emplace_back(first, first + histogram_bins[channel]);
    }
    std::vector<thrust::device_vector<SampleT>> d_levels_per_channel;
    d_levels_per_channel.emplace_back(std::move(levels_r));
    d_levels_per_channel.emplace_back(std::move(levels_g));
    d_levels_per_channel.emplace_back(std::move(levels_b));
    bench_verify_histogram_range<num_channels, num_active_channels, SampleT, CounterT, OffsetT>(
      input, opt_hists_d, d_levels_per_channel, static_cast<OffsetT>(elements), "multi.range");
    levels_r   = std::move(d_levels_per_channel[0]);
    levels_g   = std::move(d_levels_per_channel[1]);
    levels_b   = std::move(d_levels_per_channel[2]);
    d_levels_r = thrust::raw_pointer_cast(levels_r.data());
    d_levels_g = thrust::raw_pointer_cast(levels_g.data());
    d_levels_b = thrust::raw_pointer_cast(levels_b.data());
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
                 d_histograms,
                 cuda::std::array<int, num_active_channels>{num_levels_r, num_levels_g, num_levels_b},
                 cuda::std::array<const SampleT*, num_active_channels>{d_levels_r, d_levels_g, d_levels_b},
                 static_cast<OffsetT>(elements),
                 env);
               timer.stop();
             });
}

// CounterT / OffsetT overridable for the 64-bit-counter / 64-bit-offset variant
// (matches the single-channel benches). Inert when undefined. Multi-channel 8-byte
// counters are the case where the multi on-chip SMEM caps must be byte-derived.
#ifdef TUNE_CounterT
using counter_types = nvbench::type_list<TUNE_CounterT>;
#else
using counter_types = nvbench::type_list<int32_t>;
#endif
#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t>;
#endif

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
  // One `concentrated` shape swept across entropy (1.0=uniform, 0.5=spike,
  // 0.0=constant) plus the multi-hot and cache-adversarial shapes. Each value
  // may carry an inline knob as "name:value"; see histogram_inputs.cuh.
  .add_string_axis(
    "InputShape",
    {"concentrated:1.0",
     "concentrated:0.5",
     "concentrated:0.0",
     "powerlaw:0.5",
     "hash_synonym",
     "stale_resident:0.5",
     "stale_resident:0.25",
     "temporal_phases:0.10",
     "strided_sweep",
     "sawtooth"});
