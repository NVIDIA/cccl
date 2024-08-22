// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_ALGORITHM alg 0:4:1

// keep checks at the top so compilation of discarded variants fails really fast
#if !TUNE_BASE
#  if TUNE_ALGORITHM == 4 && (__CUDA_ARCH_LIST__) < 900
#    error "Cannot compile algorithm 4 (ublkcp) below sm90"
#  endif

#  if TUNE_ALGORITHM != 2 && TUNE_ITEMS_PER_THREAD != 1
#    error "Skip ITEMS_PER_THREAD tuning for algorithms other than 2"
#  endif

#  if TUNE_ALGORITHM == 4 && !defined(_CUB_HAS_TRANSFORM_UBLKCP)
#    error "Cannot tune for ublkcp algorithm, which is not provided by CUB (old CTK?)"
#  endif

#endif

#include "babelstream.h"

#if !TUNE_BASE
#  if CUB_DETAIL_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "This benchmark does not support being compiled for multiple architectures"
#  endif
#endif

template <typename T, typename OffsetT>
static void mul(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(state, ::cuda::std::tuple{c.begin()}, b.begin(), n, [=] _CCCL_DEVICE(const T& ci) {
    return ci * scalar;
  });
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

#if 0
// TODO(bgruber): I guess we should split this into a seperate file? But it should be tuned together with the rest here.
static void heavy(nvbench::state& state)
{
  using T                  = uint32_t; // must be unsigned so overflow is defined
  constexpr auto heavyness = 128;

  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in(n, startA);
  thrust::device_vector<T> out(n, startB);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), n, [=] _CCCL_DEVICE(T data) {
    T reg[heavyness];
    reg[0] = data;
    for (int i = 1; i < heavyness; ++i)
    {
      reg[i] = reg[i - 1] * reg[i - 1] + 1;
    }
    for (int i = 0; i < heavyness; ++i)
    {
      reg[i] = (reg[i] * reg[i]) % 19;
    }
    for (int i = 0; i < heavyness; ++i)
    {
      reg[i] = reg[heavyness - i - 1] * reg[i];
    }
    T x = 0;
    for (int i = 0; i < heavyness; ++i)
    {
      x += reg[i];
    }
    return x;
  });
}

// TODO(bgruber): search.py fails because we don't have the same type axes as the other benchmarks. But I don't need different types here.
NVBENCH_BENCH(heavy).set_name("heavy").add_int64_power_of_two_axis("Elements{io}", array_size_powers);
#endif
