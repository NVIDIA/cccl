//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/random>
#include <cuda/stream>

#include "nvbench_helper.cuh"

NVBENCH_DECLARE_TYPE_STRINGS(cuda::std::minstd_rand, "minstd", "cuda::std::minstd_rand");
NVBENCH_DECLARE_TYPE_STRINGS(cuda::std::philox4x32, "philox", "cuda::std::philox4x32");

using rng_types = nvbench::type_list<cuda::std::minstd_rand, cuda::std::philox4x32>;

template <class Rng>
__global__ void cuda_sample_kernel(
  const int* __restrict__ population, cuda::std::size_t n, cuda::std::size_t k, Rng rng, int* __restrict__ out)
{
  const auto* end = cuda::sample(population, population + n, out, k, rng);
  // So nvcc doesn't optimize the sampling away
  out[0] += static_cast<int>(end - out - k);
}

template <class Rng>
__global__ void cuda_std_sample_kernel(
  const int* __restrict__ population, cuda::std::size_t n, cuda::std::size_t k, Rng rng, int* __restrict__ out)
{
  const auto* end = cuda::std::sample(population, population + n, out, k, rng);
  // So nvcc doesn't optimize the sampling away
  out[0] += static_cast<int>(end - out - k);
}

template <class Rng, class Kernel>
void run(
  nvbench::state& state, Kernel kernel, cuda::std::size_t n, cuda::std::size_t k, cuda::std::size_t population_reads)
{
  const auto device = cuda::device_ref{state.get_device()->get_id()}; // NOLINT(bugprone-unchecked-optional-access)
  auto stream       = cuda::stream{device};
  auto resource     = cuda::device_default_memory_pool(device);
  auto population   = cuda::make_buffer<int>(stream, resource, n, 0);
  auto out          = cuda::make_buffer<int>(stream, resource, k, 0);
  // The fills run on `stream`, the timed kernels run on the nvbench stream.
  stream.sync();

  state.add_element_count(k);
  // `k` for `cuda::sample`, `n` for `cuda::std::sample`, but not more than n
  state.add_global_memory_reads<int>(cuda::std::min(population_reads, n));
  state.add_global_memory_writes<int>(cuda::std::min(k, n));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<1, 1, 0, launch.get_stream()>>>(population.data(), n, k, Rng{42}, out.data());
  });
}

template <class Rng>
void cuda_sample(nvbench::state& state, nvbench::type_list<Rng>)
{
  const auto n = static_cast<cuda::std::size_t>(state.get_int64("PopulationSize"));
  const auto k = static_cast<cuda::std::size_t>(state.get_int64("SampleSize"));

  run<Rng>(state, cuda_sample_kernel<Rng>, n, k, k);
}

NVBENCH_BENCH_TYPES(cuda_sample, NVBENCH_TYPE_AXES(rng_types))
  .set_name("cuda_sample")
  .set_type_axes_names({"Rng{ct}"})
  .add_int64_power_of_two_axis("PopulationSize", nvbench::range(10, 22, 4))
  .add_int64_power_of_two_axis("SampleSize", {6, 12});

template <class Rng>
void cuda_std_sample(nvbench::state& state, nvbench::type_list<Rng>)
{
  const auto n = static_cast<cuda::std::size_t>(state.get_int64("PopulationSize"));
  const auto k = static_cast<cuda::std::size_t>(state.get_int64("SampleSize"));

  run<Rng>(state, cuda_std_sample_kernel<Rng>, n, k, n);
}

NVBENCH_BENCH_TYPES(cuda_std_sample, NVBENCH_TYPE_AXES(rng_types))
  .set_name("std_sample")
  .set_type_axes_names({"Rng{ct}"})
  .add_int64_power_of_two_axis("PopulationSize", nvbench::range(10, 22, 4))
  .add_int64_power_of_two_axis("SampleSize", {6, 12});
