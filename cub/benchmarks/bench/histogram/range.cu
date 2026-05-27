// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/host_vector.h>

#include <nvbench_helper.cuh>

#include "histogram_common.cuh"

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
  const auto entropy   = str_to_entropy(state.get_string("Entropy"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  // Quadratic spacing keeps DispatchRange on the SearchTransform path.
  thrust::host_vector<SampleT> h_levels(num_bins + 1);
  h_levels[0]    = lower_level;
  const double L = static_cast<double>(lower_level);
  const double U = static_cast<double>(upper_level);
  const double n = static_cast<double>(num_bins);
  for (int i = 1; i <= num_bins; ++i)
  {
    const double t = static_cast<double>(i) / n;
    SampleT lvl    = static_cast<SampleT>(L + (U - L) * t * t);
    if (lvl <= h_levels[i - 1])
    {
      lvl = static_cast<SampleT>(h_levels[i - 1] + SampleT{1});
    }
    h_levels[i] = lvl;
  }
  thrust::device_vector<SampleT> levels = h_levels;
  SampleT* d_levels                     = thrust::raw_pointer_cast(levels.data());

  thrust::device_vector<SampleT> input = generate(elements, entropy, lower_level, upper_level);
  thrust::device_vector<CounterT> hist(num_bins);

  SampleT* d_input      = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

  caching_allocator_t alloc;
  // Demote persisting L2 lines outside the timed window so that no
  // cudaAccessPolicyWindow set by the dispatcher can carry across iterations.
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cudaCtxResetPersistingL2Cache();
               timer.start();
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
  .add_int64_axis("Bins", {32, 128, 2048, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"});
