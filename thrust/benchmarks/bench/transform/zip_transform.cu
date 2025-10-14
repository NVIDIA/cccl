// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

#include <cuda/__functional/address_stability.h>

#include <nvbench_helper.cuh>

// This benchmark is intended to be compared to the nstream benchmark from babelstream.cu, so we can:
// * detect regressions in the unpacking of a zip_transform_iterator

// same variables as in basic.cu so we can compare results
constexpr auto startA      = 1; // BabelStream: 0.1
constexpr auto startB      = 2; // BabelStream: 0.2
constexpr auto startC      = 3; // BabelStream: 0.1
constexpr auto startScalar = 4; // BabelStream: 0.4
using element_types        = nvbench::type_list<std::int8_t, std::int16_t, float, double, __int128>;
auto array_size_powers     = std::vector<std::int64_t>{25, 31};

template <typename T>
static void nstream_zip_transform(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto lambda    = cuda::proclaim_copyable_arguments([scalar] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) -> T {
    return ai + bi + scalar * ci;
  });
  cuda::zip_transform_iterator begin{lambda, a.begin(), b.begin(), c.begin()};
  cuda::zip_transform_iterator end{lambda, a.end(), b.end(), c.end()};
  caching_allocator_t alloc; // transform shouldn't allocate, but let's be consistent
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::copy(policy(alloc, launch), begin, end, a.begin());
             });
}

NVBENCH_BENCH_TYPES(nstream_zip_transform, NVBENCH_TYPE_AXES(element_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);
