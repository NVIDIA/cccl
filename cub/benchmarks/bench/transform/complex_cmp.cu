// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_ALGORITHM alg 0:4:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// for TUNE_ALGORITHM == 1 (vectorized), this is the number of vectors per thread, which is similar in spirit
// %RANGE% TUNE_UNROLL_FACTOR unrl 1:4:1

// those parameters only apply if TUNE_ALGORITHM == 0 (prefetch)
// %RANGE% TUNE_PREFETCH_MULT pref 1:3:1

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1

#if !TUNE_BASE && TUNE_ALGORITHM != 0 && (TUNE_PREFETCH_MULT != 1)
#  error "Non-prefetch algorithms require prefetch multiple to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 0 && (TUNE_PREFETCH_MULT != 1)

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1)
#  error "Non-vectorized algorithms require vector size to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1)

#include "common.h"

// This benchmark tests overlapping memory regions for reading and is compute intensive

static void compare_complex(nvbench::state& state)
try
{
  const auto n                        = state.get_int64("Elements{io}");
  thrust::device_vector<complex32> in = generate(n);
  thrust::device_vector<bool> out(n - 1);

  state.add_element_count(n);
  state.add_global_memory_reads<complex32>(n);
  state.add_global_memory_writes<bool>(n);

  // the complex comparison needs lots of compute and transform reads from overlapping input
  using compare_op = less_t;
  bench_transform(state, cuda::std::tuple{in.begin(), in.begin() + 1}, out.begin(), n - 1, compare_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(compare_complex)
  .set_name("compare_complex")
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
