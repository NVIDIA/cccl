// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// for filling, we can only use the prefetch and the vectorized algorithm
// %RANGE% TUNE_ALGORITHM alg 0:2:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// those parameters only apply if TUNE_ALGORITHM == 0 (prefetch)
// %RANGE% TUNE_ITEMS_PER_THREAD_NO_INPUT ipt 1:32:1

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1
// %RANGE% TUNE_VECTORS_PER_THREAD vpt 1:4:1

#if !TUNE_BASE && TUNE_ALGORITHM != 0 && (TUNE_ITEMS_PER_THREAD_NO_INPUT != 1)
#  error "Non-prefetch algorithms require the no input items per thread to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)
#  error "Non-vectorized algorithms require vector size and vectors per thread to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)

#include "common.h"

template <typename T>
struct return_constant
{
  T value;

  _CCCL_DEVICE auto operator()() const -> T
  {
    return value;
  }
};

template <typename T>
static void fill(nvbench::state& state, nvbench::type_list<T>)
try
{
  // A 32-bit offset type or the value 0 or 0xFF... have <1% performance impact
  using offset_t   = int64_t;
  const auto value = T{42};
  const auto n     = state.get_int64("Elements{io}");
  thrust::device_vector<T> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(0);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, ::cuda::std::tuple{}, out.begin(), cuda::narrow<offset_t>(n), return_constant<T>{value});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(fill, NVBENCH_TYPE_AXES(integral_types))
  .set_name("fill")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
