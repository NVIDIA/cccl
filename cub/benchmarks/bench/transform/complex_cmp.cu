// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

// keep checks at the top so compilation of discarded variants fails really fast
#if !TUNE_BASE
#  if TUNE_ALGORITHM == 1 && (__CUDA_ARCH_LIST__) < 900
#    error "Cannot compile algorithm 4 (ublkcp) below sm90"
#  endif

#  if TUNE_ALGORITHM == 1 && !defined(_CUB_HAS_TRANSFORM_UBLKCP)
#    error "Cannot tune for ublkcp algorithm, which is not provided by CUB (old CTK?)"
#  endif
#endif

#include "common.h"

#if !TUNE_BASE
#  if CUB_DETAIL_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "This benchmark does not support being compiled for multiple architectures"
#  endif
#endif

// This benchmark tests overlapping memory regions for reading and is compute intensive

template <typename OffsetT>
static void compare_complex(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto n                      = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<complex> in = generate(n);
  thrust::device_vector<bool> out(n - 1);

  state.add_element_count(n);
  state.add_global_memory_reads<complex>(n);
  state.add_global_memory_writes<bool>(n);

  // the complex comparison needs lots of compute and transform reads from overlapping input
  using compare_op = less_t;
  bench_transform(state, ::cuda::std::tuple{in.begin(), in.begin() + 1}, out.begin(), n - 1, compare_op{});
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(compare_complex, NVBENCH_TYPE_AXES(offset_types))
  .set_name("compare_complex")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
