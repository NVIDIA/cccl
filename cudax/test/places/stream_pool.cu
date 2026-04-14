//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Tests for the standalone stream pool functionality in exec_place.
 *
 * Verifies that exec_place::pick_stream() works without a CUDASTF context
 * or async_resources_handle, returning valid CUDA streams from the
 * per-place stream pool.
 */

#include <cuda/experimental/__places/places.cuh>

using namespace cuda::experimental::places;

__global__ void increment_kernel(int* data, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    data[tid] += 1;
  }
}

// Streams returned by pick_stream() are owned by the exec_place's internal
// pool (round-robin, lazily created). Callers must NOT destroy them.
void test_basic_pick_stream()
{
  exec_place place = exec_place::current_device();

  cudaStream_t stream = place.pick_stream();
  _CCCL_ASSERT(stream != nullptr, "pick_stream must return a valid stream");

  int current_device;
  cuda_try(cudaGetDevice(&current_device));
  _CCCL_ASSERT(get_device_from_stream(stream) == current_device, "stream must belong to the current device");

  fprintf(stderr, "test_basic_pick_stream: PASSED\n");
}

void test_pick_stream_computation_hint()
{
  exec_place place = exec_place::current_device();

  cudaStream_t compute_stream  = place.pick_stream(true);
  cudaStream_t transfer_stream = place.pick_stream(false);

  _CCCL_ASSERT(compute_stream != nullptr, "compute stream must be valid");
  _CCCL_ASSERT(transfer_stream != nullptr, "transfer stream must be valid");

  fprintf(stderr, "test_pick_stream_computation_hint: PASSED\n");
}

void test_pick_stream_specific_device(int ndevs)
{
  if (ndevs < 2)
  {
    fprintf(stderr, "test_pick_stream_specific_device: skipped (need >= 2 devices)\n");
    return;
  }

  for (int d = 0; d < ndevs && d < 2; d++)
  {
    exec_place dev      = exec_place::device(d);
    cudaStream_t stream = dev.pick_stream();
    _CCCL_ASSERT(stream != nullptr, "stream must be valid");
    _CCCL_ASSERT(get_device_from_stream(stream) == d, "stream must belong to the requested device");
  }

  fprintf(stderr, "test_pick_stream_specific_device: PASSED\n");
}

void test_launch_kernel_on_picked_stream()
{
  exec_place place    = exec_place::current_device();
  cudaStream_t stream = place.pick_stream();

  constexpr int N = 256;
  int* d_data;
  cuda_try(cudaMallocAsync(&d_data, N * sizeof(int), stream));
  cuda_try(cudaMemsetAsync(d_data, 0, N * sizeof(int), stream));

  increment_kernel<<<1, N, 0, stream>>>(d_data, N);

  int h_data[N];
  cuda_try(cudaMemcpyAsync(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_try(cudaStreamSynchronize(stream));

  for (int i = 0; i < N; i++)
  {
    _CCCL_ASSERT(h_data[i] == 1, "kernel result mismatch");
  }

  cuda_try(cudaFreeAsync(d_data, stream));
  cuda_try(cudaStreamSynchronize(stream));

  fprintf(stderr, "test_launch_kernel_on_picked_stream: PASSED\n");
}

void test_round_robin_streams()
{
  exec_place place = exec_place::current_device();

  cudaStream_t first  = place.pick_stream();
  cudaStream_t second = place.pick_stream();

  _CCCL_ASSERT(first != nullptr, "first stream must be valid");
  _CCCL_ASSERT(second != nullptr, "second stream must be valid");

  fprintf(stderr, "test_round_robin_streams: PASSED\n");
}

int main()
{
  int ndevs;
  cuda_try(cudaGetDeviceCount(&ndevs));

  test_basic_pick_stream();
  test_pick_stream_computation_hint();
  test_pick_stream_specific_device(ndevs);
  test_launch_kernel_on_picked_stream();
  test_round_robin_streams();
}
