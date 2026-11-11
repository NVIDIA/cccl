//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/random>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>

#include <nvbench/nvbench.cuh>

using cuda::experimental::mwc128x32;
using cuda::experimental::mwc64x32;

NVBENCH_DECLARE_TYPE_STRINGS(mwc64x32, "mwc64x32", "cuda::experimental::mwc64x32");
NVBENCH_DECLARE_TYPE_STRINGS(mwc128x32, "mwc128x32", "cuda::experimental::mwc128x32");

using rng_types = nvbench::type_list<mwc64x32, mwc128x32>;

template <class Engine>
__global__ void initialize(Engine* states, const uint64_t seed)
{
  const auto index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  states[index]    = Engine{seed + index};
}

template <class Engine>
__global__ void generate(const Engine* initial_states, const uint32_t samples_per_thread)
{
  const auto index  = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto engine       = initial_states[index];
  uint32_t checksum = 0;
  for (uint32_t i = 0; i < samples_per_thread; ++i)
  {
    checksum += engine();
  }
  // Keep the scrambler and every state transition observable without per-sample stores.
  sink(cuda::std::pair{engine, checksum});
}

template <class Engine>
void generation(nvbench::state& state, nvbench::type_list<Engine>)
{
  const auto requested_samples  = state.get_int64("SamplesPerThread");
  const auto samples_per_thread = static_cast<uint32_t>(requested_samples);
  const auto block_size         = static_cast<int>(state.get_int64("BlockSize"));
  int blocks_per_sm             = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, generate<Engine>, block_size, 0));
  if (blocks_per_sm == 0)
  {
    state.skip("Insufficient resources for the requested block size");
    return;
  }
  const int num_sms = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  // Launch one resident wave for this engine and block size.
  const int grid_size = blocks_per_sm * num_sms;
  const auto threads  = static_cast<size_t>(grid_size) * block_size;
  thrust::device_vector<Engine> initial_states(threads);
  auto* const initial = thrust::raw_pointer_cast(initial_states.data());

  // Seed once outside timing; every sample starts from the same per-thread state.
  initialize<<<grid_size, block_size>>>(initial, 42);
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaGetLastError());
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaDeviceSynchronize());

  state.add_element_count(static_cast<uint64_t>(threads) * samples_per_thread);
  state.add_global_memory_reads<Engine>(threads);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    generate<<<grid_size, block_size, 0, launch.get_stream()>>>(initial, samples_per_thread);
  });
}

NVBENCH_BENCH_TYPES(generation, NVBENCH_TYPE_AXES(rng_types))
  .set_name("generation")
  .set_type_axes_names({"Engine{ct}"})
  .add_int64_axis("BlockSize", {128, 256, 512, 1024})
  .add_int64_power_of_two_axis("SamplesPerThread", {16});
