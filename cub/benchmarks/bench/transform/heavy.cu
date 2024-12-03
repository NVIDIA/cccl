// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

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
{
  using value_t                     = std::uint32_t;
  using offset_t                    = int;
  const auto n                      = narrow<offset_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<value_t> in = generate(n);
  thrust::device_vector<value_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<value_t>(n);

  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), n, heavy_functor<Heaviness::value>{});
}

template <int I>
using ic = ::cuda::std::integral_constant<int, I>;

NVBENCH_BENCH_TYPES(heavy, NVBENCH_TYPE_AXES(nvbench::type_list<ic<32>, ic<64>, ic<128>, ic<256>>))
  .set_name("heavy")
  .set_type_axes_names({"Heaviness{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
