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

// This benchmark overwrites it inputs, has a uniform workload and is compute intensive

struct Transform
{
  float mat[3][4];

  __device__ auto operator()(float3 v) const -> float3
  {
    float3 r;
    r.x = mat[0][0] * v.x + mat[0][1] * v.y + mat[0][2] * v.z + mat[0][3];
    r.y = mat[1][0] * v.x + mat[1][1] * v.y + mat[1][2] * v.z + mat[1][3];
    r.z = mat[2][0] * v.x + mat[2][1] * v.y + mat[2][2] * v.z + mat[2][3];
    return r;
  }
};

template <typename OffsetT>
static void vertex_transform(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<float3> data(n, float3{1, 2, 3}); // generate(n); does not work for float3
  const auto transform = Transform{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}};

  state.add_element_count(n);
  state.add_global_memory_reads<float3>(n);
  state.add_global_memory_writes<float3>(n);

  bench_transform(state, ::cuda::std::tuple{data.begin()}, data.begin(), n, transform);
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(vertex_transform, NVBENCH_TYPE_AXES(offset_types))
  .set_name("vertex_transform")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);
