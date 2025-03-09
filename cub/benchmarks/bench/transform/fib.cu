// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

// This benchmark is compute intensive with diverging threads

template <class IndexT, class OutputT>
struct fib_t
{
  __device__ OutputT operator()(IndexT n)
  {
    OutputT t1 = 0;
    OutputT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    if (n == 1)
    {
      return t1;
    }
    if (n == 2)
    {
      return t2;
    }
    for (IndexT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }
    return t2;
  }
};
template <typename OffsetT>
static void fibonacci(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t                     = int64_t;
  using output_t                    = uint32_t;
  const auto n                      = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<index_t> in = generate(n, bit_entropy::_1_000, index_t{0}, index_t{42});
  thrust::device_vector<output_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<index_t>(n);
  state.add_global_memory_writes<output_t>(n);

  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), n, fib_t<index_t, output_t>{});
}

NVBENCH_BENCH_TYPES(fibonacci, NVBENCH_TYPE_AXES(offset_types))
  .set_name("fibonacci")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
