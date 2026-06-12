// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Tile variant of the BabelStream transform bench. The lambdas of the base benchmark are replaced by
// named, stateless ops that register a tile_operator substitute (gated). Under --enable-tile +
// CCCL_ENABLE_TILE_TRANSFORM_DISPATCH the dispatch hook routes them to the tile kernel; otherwise this
// is the standard CUB transform path. This file disappears once tile dispatch is fully transparent.

#include "../common.h"

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

// Stateless scalar ops, used at the call site in both build modes. Constants are baked in so the ops
// stay stateless (the tile substitute must be trivially default constructible): with startScalar == -2,
// `c * scalar` is `-(c + c)`, `b + scalar * c` is `b - c - c`, etc.
struct mul_op
{
  template <class B>
  __host__ __device__ auto operator()(B b) const
  {
    return -(b + b);
  }
};
struct add_op
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a + b;
  }
};
struct triad_op
{
  template <class B, class C>
  __host__ __device__ auto operator()(B b, C c) const
  {
    return b - c - c;
  }
};
struct nstream_op
{
  template <class A, class B, class C>
  __host__ __device__ auto operator()(A a, B b, C c) const
  {
    return a + b - c - c;
  }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
struct tile_mul_op
{
  template <class B>
  __tile__ auto operator()(B b) const
  {
    return -(b + b);
  }
};
struct tile_add_op
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a + b;
  }
};
struct tile_triad_op
{
  template <class B, class C>
  __tile__ auto operator()(B b, C c) const
  {
    return b - c - c;
  }
};
struct tile_nstream_op
{
  template <class A, class B, class C>
  __tile__ auto operator()(A a, B b, C c) const
  {
    return a + b - c - c;
  }
};

CUB_NAMESPACE_BEGIN
namespace transform
{
template <class T>
inline constexpr bool tile_eligible_v<mul_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<add_op, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<triad_op, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<nstream_op, T, 3> = true;
template <>
struct tile_operator<mul_op>
{
  using type = tile_mul_op;
};
template <>
struct tile_operator<add_op>
{
  using type = tile_add_op;
};
template <>
struct tile_operator<triad_op>
{
  using type = tile_triad_op;
};
template <>
struct tile_operator<nstream_op>
{
  using type = tile_nstream_op;
};
} // namespace transform
CUB_NAMESPACE_END
#endif // _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()

// The tile path does not support __int128 (no tensor_span/partition_view for it), so the type axis
// omits it relative to the base babelstream bench.
#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types = nvbench::type_list<std::int8_t, std::int16_t, float, double>;
#endif

inline auto array_size_powers = nvbench::range(16, 32, 4);

// Same constant inputs as the base bench so nstream maintains a consistent workload.
inline constexpr auto startA      = 11;
inline constexpr auto startB      = 2;
inline constexpr auto startC      = 1;
inline constexpr auto startScalar = -2;
static_assert(startA == (startA + startB + startScalar * startC), "nstream must have a consistent workload");

template <typename T>
static void mul(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{c.begin()}, b.begin(), n, mul_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types))
  .set_name("tile_mul")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void add(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{a.begin(), b.begin()}, c.begin(), n, add_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types))
  .set_name("tile_add")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void triad(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{b.begin(), c.begin()}, a.begin(), n, triad_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types))
  .set_name("tile_triad")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void nstream(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{a.begin(), b.begin(), c.begin()}, a.begin(), n, nstream_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types))
  .set_name("tile_nstream")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);
