// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

template <typename T, typename OffsetT>
static void nstream(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n         = narrow<OffsetT>(state.get_int64("Elements{io}"));
  const auto overwrite = static_cast<bool>(state.get_int64("OverwriteInput"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  // The BabelStream nstream overwrites one input array to avoid write-allocation of cache lines. However, this changes
  // the data that is computed for each iteration and results in an unstable workload. Therefore, we added an axis to
  // choose a different output array. Pass `-a OverwriteInput=0` to the benchmark to disable overwriting the input.
  thrust::device_vector<T> d;
  if (!overwrite)
  {
    d.resize(n);
  }

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state,
    ::cuda::std::tuple{a.begin(), b.begin(), c.begin()},
    overwrite ? a.begin() : d.begin(),
    n,
    [=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
      return ai + bi + scalar * ci;
    },
    nvbench::exec_tag::none); // Use batch mode for benchmarking since the workload changes. Not necessary when
                              // OverwriteInput=0, but doesn't hurt
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers)
  .add_int64_axis("OverwriteInput", {1});
