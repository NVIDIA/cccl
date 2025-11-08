// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/__functional/address_stability.h>

#include <nvbench_helper.cuh>

// The benchmarks are inspired by the BabelStream thrust version:
// https://github.com/UoB-HPC/BabelStream/blob/main/src/thrust/ThrustStream.cu

// Modified from BabelStream to also work for integers
constexpr auto startA      = 1; // BabelStream: 0.1
constexpr auto startB      = 2; // BabelStream: 0.2
constexpr auto startC      = 3; // BabelStream: 0.1
constexpr auto startScalar = 4; // BabelStream: 0.4

using element_types = nvbench::type_list<std::int8_t, std::int16_t, float, double, __int128>;
// Different benchmarks use a different number of buffers. H200/B200 can fit 2^31 elements for all benchmarks and types.
// Upstream BabelStream uses 2^25. Allocation failure just skips the benchmark
auto array_size_powers = std::vector<std::int64_t>{25, 31};

template <typename... Args>
void bench_transform(nvbench::state& state, Args&&... args)
{
  caching_allocator_t alloc; // transform shouldn't allocate, but let's be consistent
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::transform(policy(alloc, launch), ::cuda::std::forward<Args>(args)...);
             });
}

template <typename T>
static void mul(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(
    state, c.begin(), c.end(), b.begin(), cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& ci) {
      return ci * scalar;
    }));
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void add(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  bench_transform(
    state,
    a.begin(),
    a.end(),
    b.begin(),
    c.begin(),
    cuda::proclaim_copyable_arguments([] _CCCL_DEVICE(const T& ai, const T& bi) -> T {
      return ai + bi;
    }));
}

NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types))
  .set_name("add")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void triad(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(
    state,
    b.begin(),
    b.end(),
    c.begin(),
    a.begin(),
    cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& bi, const T& ci) {
      return bi + scalar * ci;
    }));
}

NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types))
  .set_name("triad")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void nstream(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(
    state,
    thrust::make_zip_iterator(a.begin(), b.begin(), c.begin()),
    thrust::make_zip_iterator(a.end(), b.end(), c.end()),
    a.begin(),
    thrust::make_zip_function(cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
      return ai + bi + scalar * ci;
    })));
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

// variation of nstream requiring a stable parameter address because it recovers the element index
template <typename T>
static void nstream_stable(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  const T* a_start = thrust::raw_pointer_cast(a.data());
  const T* b_start = thrust::raw_pointer_cast(b.data());
  const T* c_start = thrust::raw_pointer_cast(c.data());

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(state, a.begin(), a.end(), a.begin(), [=] _CCCL_DEVICE(const T& ai) {
    const auto i = &ai - a_start;
    return ai + b_start[i] + scalar * c_start[i];
  });
}

NVBENCH_BENCH_TYPES(nstream_stable, NVBENCH_TYPE_AXES(element_types))
  .set_name("nstream_stable")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);
