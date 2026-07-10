// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

#include "histogram_common.cuh"
#include "histogram_inputs.cuh"

// %RANGE% TUNE_ITEMS ipt 4:28:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_LOAD_ALGORITHM_ID laid 0:2:1
// %RANGE% TUNE_VEC_SIZE_POW vec 0:2:1

template <typename SampleT, typename LocalCounterT, typename GlobalCounterT, typename OffsetT>
static void even(nvbench::state& state, nvbench::type_list<SampleT, LocalCounterT, GlobalCounterT, OffsetT>)
{
  const auto shape     = parse_input_shape(state.get_string("InputShape"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;

  // Skip invalid configurations where the SampleT range can't hold enough
  // strictly-monotonic levels: bins + 1 levels require bins + 1 distinct
  // SampleT values, and the bench's `[get_lower_level, get_upper_level]`
  // range spans at most `max_representable_bins<SampleT>() + 1` distinct
  // values.
  if (num_bins > max_representable_bins<SampleT>())
  {
    state.skip("Number of bins exceeds what SampleT can represent");
    return;
  }

  const SampleT lower_level = get_lower_level<SampleT>();
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  // Per-block direct-atomic SMEM cache slot count S, queried from CUB's occupancy
  // sizer so hash_synonym, stale_resident, and poison track the ACTUAL cache.
  // Single-channel EVEN path. Cache capacity is governed by the local counter,
  // while the output allocation uses GlobalCounterT.
  // The byte extent selects the int vs wide-OffsetT kernel the dispatch will launch at
  // this N (row_stride_bytes == sizeof(SampleT) * elements for a single channel).
#if defined(CUB_HISTO_BENCH_DISABLE_CACHE_QUERY)
  // An overlaid stock-main baseline has no direct-atomic cache query. The sweep
  // supplies the branch's queried value through CUB_HISTO_INPUT_CACHE_SLOTS.
  const int64_t cache_slots = 0;
#else
  const int64_t cache_slots = cub::detail::histogram::query_direct_atomic_cache_slots_for_extent<
    1,
    1,
    /*IsEven=*/true,
    SampleT,
    LocalCounterT,
    SampleT,
    OffsetT,
    GlobalCounterT>(static_cast<unsigned long long>(sizeof(SampleT)) * static_cast<unsigned long long>(elements));
#endif
  bench_log_input_cache_slots(cache_slots);
  if (bench_input_cache_slot_query_only())
  {
    state.skip("input cache-slot query only");
    return;
  }

  thrust::device_vector<SampleT> input = generate_histogram_input_even<SampleT>(
    shape, elements, static_cast<int>(num_bins), lower_level, upper_level, /*seed=*/42, cache_slots);
  thrust::device_vector<GlobalCounterT> hist(num_bins);

  SampleT* d_input            = thrust::raw_pointer_cast(input.data());
  GlobalCounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<GlobalCounterT>(num_bins);

  // Warmup + correctness check: run HistogramEven once outside `state.exec`,
  // checking the dispatch return code, then verify that the histogram sums
  // to the input sample count and matches an independent reference bin-by-bin.
  // A failure here aborts before any timed iteration runs, so a silent dispatch
  // failure, counter overflow, or sample-loss bug can't inflate the measured
  // bandwidth. Skipped when
  // CUB_BENCH_HISTOGRAM_VERIFY=0|false|no|off.
  if (bench_correctness_checks_enabled())
  {
    thrust::fill(hist.begin(), hist.end(), GlobalCounterT{0});
    bench_check_cuda(
      cub::DeviceHistogram::HistogramEven(
        d_input,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        static_cast<OffsetT>(elements),
        cuda::execution::tune(bench_policy_selector<SampleT, LocalCounterT, 1, 1, true>{})),
      "warmup HistogramEven");
    bench_check_cuda(cudaDeviceSynchronize(), "warmup sync");

    std::vector<thrust::device_vector<GlobalCounterT>> opt_hists_d;
    opt_hists_d.emplace_back(std::move(hist));
    bench_verify_histogram_even<1, 1, SampleT, GlobalCounterT, OffsetT>(
      input, opt_hists_d, static_cast<OffsetT>(elements), static_cast<int>(num_bins), lower_level, upper_level, "even");
    hist        = std::move(opt_hists_d[0]);
    d_histogram = thrust::raw_pointer_cast(hist.data());
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
                 alloc, launch, cuda::execution::tune(bench_policy_selector<SampleT, LocalCounterT, 1, 1, true>{}));
               _CCCL_TRY_CUDA_API(
                 cub::DeviceHistogram::HistogramEven,
                 "HistogramEven failed",
                 d_input,
                 d_histogram,
                 num_levels,
                 lower_level,
                 upper_level,
                 static_cast<OffsetT>(elements),
                 env);
               timer.stop();
             });
}

// Local and global counter widths are independent axes. TUNE_CounterT remains a
// compatibility fallback that selects the historical same-width configuration.
// A 64-bit OffsetT is required for >2^31 elements.
#ifdef TUNE_LocalCounterT
using local_counter_types = nvbench::type_list<TUNE_LocalCounterT>;
#elif defined(TUNE_CounterT)
using local_counter_types = nvbench::type_list<TUNE_CounterT>;
#else
using local_counter_types = nvbench::type_list<int32_t>;
#endif
#ifdef TUNE_GlobalCounterT
using global_counter_types = nvbench::type_list<TUNE_GlobalCounterT>;
#elif defined(TUNE_CounterT)
using global_counter_types = nvbench::type_list<TUNE_CounterT>;
#else
using global_counter_types = nvbench::type_list<int32_t>;
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

NVBENCH_BENCH_TYPES(even, NVBENCH_TYPE_AXES(sample_types, local_counter_types, global_counter_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "LocalCounter{ct}", "GlobalCounter{ct}", "OffsetT{ct}"})
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
