//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Stream-ordered lifecycle of `sharded_array` on an EXTERNAL stream —
 *        a plain `cudaStreamCreate` stream that is not part of any
 *        `place_group` stream pool (the shape of an application's own
 *        compute stream).
 *
 * Allocation and (host-initiated) destruction enqueue on the shard streams in
 * stream order, so one external stream can carry a full
 * alloc -> write -> copy-out -> destroy-with-work-in-flight -> reallocate ->
 * write cycle with a single final synchronization: the frees land behind the
 * still-running work that reads the arrays, and reallocated memory (which the
 * pool may serve from the just-freed range) is ordered after it. Host-side
 * completion timing of the underlying pool operations is implementation- and
 * driver-defined, so this test asserts ordering and correctness only.
 *
 * Lifetime rule exercised implicitly: containers must be destroyed before the
 * streams their specs reference (destruction enqueues the frees there).
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdint>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
__global__ void write_pattern_kernel(double* p, ::std::int64_t n, double base)
{
  ::std::int64_t i            = blockIdx.x * static_cast<::std::int64_t>(blockDim.x) + threadIdx.x;
  const ::std::int64_t stride = gridDim.x * static_cast<::std::int64_t>(blockDim.x);
  for (; i < n; i += stride)
  {
    p[i] = base + static_cast<double>(i);
  }
}

__global__ void count_mismatch_kernel(const double* p, ::std::int64_t n, double base, unsigned long long* mismatches)
{
  ::std::int64_t i            = blockIdx.x * static_cast<::std::int64_t>(blockDim.x) + threadIdx.x;
  const ::std::int64_t stride = gridDim.x * static_cast<::std::int64_t>(blockDim.x);
  unsigned long long local    = 0;
  for (; i < n; i += stride)
  {
    if (p[i] != base + static_cast<double>(i))
    {
      local++;
    }
  }
  if (local)
  {
    atomicAdd(mismatches, local);
  }
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  auto group = place_group::by_locality_domains();

  constexpr ::std::int64_t n_per = 1 << 22; // 32 MiB of doubles per shard

  // External stream: unknown to the group's pools.
  cudaStream_t ext = nullptr;
  cuda_safe_call(cudaStreamCreateWithFlags(&ext, cudaStreamNonBlocking));

  ::std::vector<shard_spec> specs;
  for (size_t d = 0; d < group.size(); d++)
  {
    specs.emplace_back(static_cast<size_t>(n_per), group.place(d).affine_data_place(), group.place(d), ext);
  }
  const size_t n_shards = specs.size();

  unsigned long long* d_bad = nullptr;
  double* d_chk1            = nullptr;
  double* d_chk2            = nullptr;
  cuda_safe_call(cudaMallocAsync(&d_bad, 2 * sizeof(unsigned long long), ext));
  cuda_safe_call(cudaMemsetAsync(d_bad, 0, 2 * sizeof(unsigned long long), ext));
  cuda_safe_call(cudaMallocAsync(&d_chk1, n_shards * n_per * sizeof(double), ext));
  cuda_safe_call(cudaMallocAsync(&d_chk2, n_shards * n_per * sizeof(double), ext));

  // The whole cycle below enqueues on `ext` with no intermediate host sync.
  {
    auto arr1 = sharded_array<double>::allocate(specs);
    for (size_t d = 0; d < n_shards; d++)
    {
      write_pattern_kernel<<<256, 256, 0, ext>>>(arr1.shard(d).data, n_per, 1000.0 * static_cast<double>(d));
      cuda_safe_call(cudaGetLastError());
      cuda_safe_call(
        cudaMemcpyAsync(d_chk1 + d * n_per, arr1.shard(d).data, n_per * sizeof(double), cudaMemcpyDeviceToDevice, ext));
    }
    // Host-side destruction while the writes/copies are (potentially) still
    // in flight: the frees are enqueued on `ext` BEHIND them.
  }
  {
    // Reallocation: the pool may serve this from the just-freed range; the
    // stream ordering must keep it correct either way.
    auto arr2 = sharded_array<double>::allocate(specs);
    for (size_t d = 0; d < n_shards; d++)
    {
      write_pattern_kernel<<<256, 256, 0, ext>>>(arr2.shard(d).data, n_per, 5000.0 * static_cast<double>(d + 1));
      cuda_safe_call(cudaGetLastError());
      cuda_safe_call(
        cudaMemcpyAsync(d_chk2 + d * n_per, arr2.shard(d).data, n_per * sizeof(double), cudaMemcpyDeviceToDevice, ext));
    }
  }
  for (size_t d = 0; d < n_shards; d++)
  {
    count_mismatch_kernel<<<256, 256, 0, ext>>>(d_chk1 + d * n_per, n_per, 1000.0 * static_cast<double>(d), d_bad);
    count_mismatch_kernel<<<256, 256, 0, ext>>>(
      d_chk2 + d * n_per, n_per, 5000.0 * static_cast<double>(d + 1), d_bad + 1);
    cuda_safe_call(cudaGetLastError());
  }

  // THE one synchronization.
  cuda_safe_call(cudaStreamSynchronize(ext));

  unsigned long long h_bad[2] = {~0ull, ~0ull};
  cuda_safe_call(cudaMemcpy(h_bad, d_bad, sizeof(h_bad), cudaMemcpyDefault));
  EXPECT(h_bad[0] == 0);
  EXPECT(h_bad[1] == 0);

  cuda_safe_call(cudaFreeAsync(d_chk1, ext));
  cuda_safe_call(cudaFreeAsync(d_chk2, ext));
  cuda_safe_call(cudaFreeAsync(d_bad, ext));
  cuda_safe_call(cudaStreamSynchronize(ext));
  cuda_safe_call(cudaStreamDestroy(ext));
  return 0;
}
