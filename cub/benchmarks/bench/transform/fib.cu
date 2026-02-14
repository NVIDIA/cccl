// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_ALGORITHM alg 0:4:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1
// %RANGE% TUNE_VECTORS_PER_THREAD vpt 1:4:1

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)
#  error "Non-vectorized algorithms require vector size and vectors per thread to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)

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
try
{
  using index_t  = int64_t;
  using output_t = uint32_t;
  const auto n   = state.get_int64("Elements{io}");
  if (sizeof(OffsetT) == 4 && n > std::numeric_limits<OffsetT>::max())
  {
    state.skip("Skipping: input size exceeds 32-bit offset type capacity.");
    return;
  }

  thrust::device_vector<index_t> in = generate(n, bit_entropy::_1_000, index_t{0}, index_t{42});
  thrust::device_vector<output_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<index_t>(n);
  state.add_global_memory_writes<output_t>(n);

  bench_transform(
    state, ::cuda::std::tuple{in.begin()}, out.begin(), static_cast<OffsetT>(n), fib_t<index_t, output_t>{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(fibonacci, NVBENCH_TYPE_AXES(offset_types))
  .set_name("fibonacci")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
