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

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(fibonacci, NVBENCH_TYPE_AXES(offset_types))
  .set_name("fibonacci")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

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
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

// This benchmark overwrites it inputs, has a uniform workload and is compute intensive

struct Transform
{
  float mat[3][4];

  auto operator()(float3 v) const -> float3
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
  const auto n                       = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<float3> data = generate(n);
  const auto transform               = Transform{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}};

  state.add_element_count(n);
  state.add_global_memory_reads<float3>(n);
  state.add_global_memory_writes<float3>(n);

  bench_transform(state, ::cuda::std::tuple{data.begin()}, data.begin(), n, transform);
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(compare_complex, NVBENCH_TYPE_AXES(offset_types))
  .set_name("vertex_transform")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

// This benchmark uses a LOT of registers and is compute intensive. It was gifted by ahendriksen. It is very expensive
// to compile.

template <int N>
struct heavy_functor
{
  // we need to use an unsigned type so overflow in arithmetic wraps around
  _CCCL_HOST_DEVICE std::uint32_t operator()(std::uint32_t data) const
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

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(heavy, NVBENCH_TYPE_AXES(nvbench::type_list<ic<32>, ic<64>, ic<128>, ic<256>>))
  .set_name("heavy")
  .set_type_axes_names({"Heaviness{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);
