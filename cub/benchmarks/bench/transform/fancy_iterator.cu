// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This benchmark checks that cub::DeviceTransform unwraps fancy iterators to the underlying raw pointers, so the
// algorithm's fast paths (vectorized loads, bulk copies, ...) are still taken. It is a proof of the optimization, not a
// tuning or sweeping benchmark. Every benchmark moves the same number of bytes as a plain BabelStream kernel (see
// babelstream.h): 'triad' for 2 reads + 1 write. Performance should be identical. The element type is 8 bits wide,
// because that is where not unwrapping costs the most (e.g. ~35% on an H200).

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "common.h"

using T = std::int8_t;

static constexpr auto startA      = T{11};
static constexpr auto startB      = T{2};
static constexpr auto startC      = T{1};
static constexpr auto startScalar = T{-2};

// transform_iterators over raw pointers as operands: a linear iterator chain per input
static void triad_transform_iterator(nvbench::state& state)
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto id_b      = cuda::transform_iterator{b.begin(), [] _CCCL_HOST_DEVICE(T x) -> T {
                                         return x;
                                            }};
  auto scaled_c  = cuda::transform_iterator{c.begin(), [=] _CCCL_HOST_DEVICE(T x) -> T {
                                             return static_cast<T>(x * scalar);
                                            }};
  bench_transform(state, cuda::std::tuple{id_b, scaled_c}, a.begin(), n, [] _CCCL_DEVICE(T ib, T sc) -> T {
    return static_cast<T>(ib + sc);
  });
}

NVBENCH_BENCH(triad_transform_iterator)
  .set_name("triad_transform_iterator")
  .add_int64_power_of_two_axis("Elements{io}", {28});

// a transform_output_iterator over a raw pointer as output
static void triad_transform_output_iterator(nvbench::state& state)
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(
    state,
    cuda::std::tuple{b.begin(), c.begin()},
    cuda::make_transform_output_iterator(
      a.begin(),
      [=] _CCCL_DEVICE(T x) -> T {
        return static_cast<T>(x * scalar);
      }),
    n,
    [] _CCCL_DEVICE(T bi, T ci) -> T {
      return static_cast<T>(bi + ci);
    });
}

NVBENCH_BENCH(triad_transform_output_iterator)
  .set_name("triad_transform_output_iterator")
  .add_int64_power_of_two_axis("Elements{io}", {28});

// a single zip_iterator whose components are transform_iterators over raw pointers: a tree of iterators
static void triad_zip_transform_iterator(nvbench::state& state)
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto id_b      = cuda::transform_iterator{b.begin(), [] _CCCL_HOST_DEVICE(T x) -> T {
                                         return x;
                                            }};
  auto scaled_c  = cuda::transform_iterator{c.begin(), [=] _CCCL_HOST_DEVICE(T x) -> T {
                                             return static_cast<T>(x * scalar);
                                            }};
  auto zip       = cuda::zip_iterator{id_b, scaled_c};
  bench_transform(state, cuda::std::tuple{zip}, a.begin(), n, cuda::make_zip_function([] _CCCL_DEVICE(T ib, T sc) -> T {
                    return static_cast<T>(ib + sc);
                  }));
}

NVBENCH_BENCH(triad_zip_transform_iterator)
  .set_name("triad_zip_transform_iterator")
  .add_int64_power_of_two_axis("Elements{io}", {28});

// artificial worst case: the input is a zip_iterator of a transform_iterator of a zip_iterator (which contains a
// constant_iterator), a counting_iterator and a constant_iterator. The output is a transform_output_iterator. All of it
// should unwrap to the raw pointers a, b and out.
static void triad_nested(nvbench::state& state)
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> out(n, startA);
  thrust::device_vector<T> a(n, startB);
  thrust::device_vector<T> b(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  // the constant 1 cancels out, but the iterator is still part of the tree
  auto inner = cuda::zip_iterator{a.begin(), b.begin(), cuda::constant_iterator{T{1}}};
  auto sum   = cuda::transform_iterator{inner, cuda::make_zip_function([] _CCCL_HOST_DEVICE(T x, T y, T one) -> T {
                                        return static_cast<T>((x + y) * one);
                                        })};
  auto outer = cuda::zip_iterator{sum, cuda::counting_iterator<int>{0}, cuda::constant_iterator{startScalar}};
  bench_transform(
    state,
    cuda::std::tuple{outer},
    cuda::make_transform_output_iterator(
      out.begin(),
      [] _CCCL_DEVICE(T x) -> T {
        return x;
      }),
    n,
    cuda::make_zip_function([] _CCCL_DEVICE(T sum, int i, T k) -> T {
      return static_cast<T>((sum * k) + static_cast<T>(i & 1));
    }));
}

NVBENCH_BENCH(triad_nested).set_name("triad_nested").add_int64_power_of_two_axis("Elements{io}", {28});
