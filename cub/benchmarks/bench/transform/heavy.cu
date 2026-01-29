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

// This benchmark uses a LOT of registers and is compute intensive.

template <int N>
struct heavy_functor
{
  // we need to use an unsigned type so overflow in arithmetic wraps around
  __device__ std::uint32_t operator()(std::uint32_t data) const
  {
    std::uint32_t reg[N];
    reg[0] = data;
    for (int i = 1; i < N; ++i)
    {
      reg[i] = reg[i - 1] * reg[i - 1] + 1;
    }
    for (int i = 0; i < N; ++i)
    {
      reg[i] = (reg[i] * reg[i]) % 19;
    }
    for (int i = 0; i < N; ++i)
    {
      reg[i] = reg[N - i - 1] * reg[i];
    }
    std::uint32_t x = 0;
    for (int i = 0; i < N; ++i)
    {
      x += reg[i];
    }
    return x;
  }
};

template <typename Heaviness>
static void heavy(nvbench::state& state, nvbench::type_list<Heaviness>)
try
{
  using value_t  = std::uint32_t;
  using offset_t = int64_t;
  const auto n   = state.get_int64("Elements{io}");

  thrust::device_vector<value_t> in = generate(n);
  thrust::device_vector<value_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<value_t>(n);

  bench_transform(
    state, ::cuda::std::tuple{in.begin()}, out.begin(), cuda::narrow<offset_t>(n), heavy_functor<Heaviness::value>{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using ::cuda::std::integral_constant;
#ifdef TUNE_Heaviness
using heaviness = nvbench::type_list<TUNE_Heaviness>; // expands to "integral_constant<int, ...>"
#else
using heaviness =
  nvbench::type_list<integral_constant<int, 32>,
                     integral_constant<int, 64>,
                     integral_constant<int, 128>,
                     integral_constant<int, 256>>;
#endif

NVBENCH_BENCH_TYPES(heavy, NVBENCH_TYPE_AXES(heaviness))
  .set_name("heavy")
  .set_type_axes_names({"Heaviness{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
