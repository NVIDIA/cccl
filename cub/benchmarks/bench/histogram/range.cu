// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/sequence.h>

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

  SampleT step = (upper_level - lower_level) / num_bins;
  thrust::device_vector<SampleT> levels(num_bins + 1);

  // TODO Extract sequence to the helper TU
  thrust::sequence(levels.begin(), levels.end(), lower_level, step);
  SampleT* d_levels = thrust::raw_pointer_cast(levels.data());

  thrust::device_vector<SampleT> input = generate(elements, entropy, lower_level, upper_level);
  thrust::device_vector<CounterT> hist(num_bins);

  SampleT* d_input      = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

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
