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
 * Verifies that exec_place::pick_stream(resources) works without a CUDASTF
 * context, returning valid CUDA streams from the per-place stream pool
 * lazily created inside an `exec_place_resources` registry.
 */

#include <cuda/experimental/__places/places.cuh>

using namespace cuda::experimental::places;

__global__ void increment_kernel(int* data, int n)
{
  const int tid = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (tid < n)
  {
    data[tid] += 1;
  }
}

// Streams returned by pick_stream(resources) are owned by the supplied
// `exec_place_resources` registry (round-robin, lazily created). Callers
// must NOT destroy them; their lifetime ends with the registry.
void test_basic_pick_stream()
{
  exec_place_resources resources;
  const exec_place place = exec_place::current_device();

  cudaStream_t stream = place.pick_stream(resources);
  _CCCL_ASSERT(stream != nullptr, "pick_stream must return a valid stream");

  int current_device;
  cuda_try(cudaGetDevice(&current_device));
  _CCCL_ASSERT(get_device_from_stream(stream) == current_device, "stream must belong to the current device");

  fprintf(stderr, "test_basic_pick_stream: PASSED\n");
}

void test_pick_stream_computation_hint()
{
  exec_place_resources resources;
  const exec_place place = exec_place::current_device();

  cudaStream_t compute_stream  = place.pick_stream(resources, true);
  cudaStream_t transfer_stream = place.pick_stream(resources, false);

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

  exec_place_resources resources;
  for (int d = 0; d < ndevs && d < 2; d++)
  {
    const exec_place dev = exec_place::device(d);
    cudaStream_t stream  = dev.pick_stream(resources);
    _CCCL_ASSERT(stream != nullptr, "stream must be valid");
    _CCCL_ASSERT(get_device_from_stream(stream) == d, "stream must belong to the requested device");
  }

  fprintf(stderr, "test_pick_stream_specific_device: PASSED\n");
}

void test_launch_kernel_on_picked_stream()
{
  exec_place_resources resources;
  const exec_place place = exec_place::current_device();
  cudaStream_t stream    = place.pick_stream(resources);

  constexpr int N = 256;
  int* d_data;
  cuda_try(cudaMallocAsync(&d_data, N * sizeof(int), stream));
  cuda_try(cudaMemsetAsync(d_data, 0, N * sizeof(int), stream));

  increment_kernel<<<1, N, 0, stream>>>(d_data, N);

  int h_data[N];
  cuda_try(cudaMemcpyAsync(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_try(cudaStreamSynchronize(stream));

  for (const auto& v : h_data)
  {
    _CCCL_ASSERT(v == 1, "kernel result mismatch");
  }

  cuda_try(cudaFreeAsync(d_data, stream));
  cuda_try(cudaStreamSynchronize(stream));

  fprintf(stderr, "test_launch_kernel_on_picked_stream: PASSED\n");
}

void test_round_robin_streams()
{
  exec_place_resources resources;
  const exec_place place = exec_place::current_device();

  cudaStream_t first  = place.pick_stream(resources);
  cudaStream_t second = place.pick_stream(resources);

  _CCCL_ASSERT(first != nullptr, "first stream must be valid");
  _CCCL_ASSERT(second != nullptr, "second stream must be valid");

  fprintf(stderr, "test_round_robin_streams: PASSED\n");
}

// Two independent registries must hand out independent streams for the same
// place: this is the property that lets multiple STF contexts (or multiple
// threads with their own `async_resources_handle`) share a device without
// touching each other's stream pools.
void test_two_handles_isolation()
{
  exec_place_resources r1;
  exec_place_resources r2;
  const exec_place place = exec_place::current_device();

  cudaStream_t s1 = place.pick_stream(r1);
  cudaStream_t s2 = place.pick_stream(r2);

  _CCCL_ASSERT(s1 != nullptr && s2 != nullptr, "streams must be valid");
  _CCCL_ASSERT(s1 != s2, "different registries must own different streams");
  _CCCL_ASSERT(r1.size() == 1 && r2.size() == 1, "each registry should hold exactly one entry");

  fprintf(stderr, "test_two_handles_isolation: PASSED\n");
}

// A registry destroyed before another is created must release its CUDA
// streams; subsequent device-reset followed by a fresh registry must not
// observe any stale handles. This is the property that lets pytest sessions
// survive `cuda.bindings.driver.cuDevicePrimaryCtxReset` between tests.
void test_reset_survives_with_fresh_registry()
{
  {
    exec_place_resources resources;
    cudaStream_t stream = exec_place::current_device().pick_stream(resources);
    cuda_try(cudaStreamSynchronize(stream));
  }
  // Old registry destroyed -> its cached streams are gone -> reset is safe.
  cuda_try(cudaDeviceReset());

  exec_place_resources resources;
  cudaStream_t stream = exec_place::current_device().pick_stream(resources);
  _CCCL_ASSERT(stream != nullptr, "fresh registry must produce a valid stream after reset");
  cuda_try(cudaStreamSynchronize(stream));

  fprintf(stderr, "test_reset_survives_with_fresh_registry: PASSED\n");
}

// get_device_from_stream must answer from the stream, not from the calling
// thread's current device, and it must do so while the stream is capturing
// without invalidating the capture (the query is made from event insertion
// paths that run under capture). Device 0 stays current throughout; the
// stream is created on each of the first two devices.
void test_device_from_capturing_stream(int ndevs)
{
  const cudaStreamCaptureMode modes[] = {
    cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal, cudaStreamCaptureModeRelaxed};

  for (int d = 0; d < ndevs && d < 2; d++)
  {
    cudaStream_t stream;
    cuda_try(cudaSetDevice(d));
    cuda_try(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_try(cudaSetDevice(0));

    _CCCL_ASSERT(get_device_from_stream(stream) == d, "stream must report the device it was created on");

    for (const auto mode : modes)
    {
      cuda_try(cudaStreamBeginCapture(stream, mode));
      _CCCL_ASSERT(get_device_from_stream(stream) == d, "device query under capture must answer from the stream");

      cudaStreamCaptureStatus status;
      cuda_try(cudaStreamIsCapturing(stream, &status));
      _CCCL_ASSERT(status == cudaStreamCaptureStatusActive, "device query must not invalidate the capture");

      cudaGraph_t graph;
      cuda_try(cudaStreamEndCapture(stream, &graph));
      cuda_try(cudaGraphDestroy(graph));
    }

    cuda_try(cudaStreamDestroy(stream));
  }

  fprintf(stderr, "test_device_from_capturing_stream: PASSED\n");
}

// Streams created with cuGreenCtxStreamCreate are the case the original
// cuStreamGetCtx does not support on CUDA 12 drivers: the device must still
// be found, with and without capture, and from a thread whose current device
// is another one.
void test_device_from_green_context_stream(int ndevs)
{
#if _CCCL_CTK_AT_LEAST(12, 5)
  if (::cuda::__driver::__version_below(12, 5))
  {
    fprintf(stderr, "test_device_from_green_context_stream: skipped (driver < 12.5)\n");
    return;
  }

  for (int d = 0; d < ndevs && d < 2; d++)
  {
    CUdevice dev;
    cuda_try(cuDeviceGet(&dev, d));

    // A green context over one group of SMs. Only CUDA_ERROR_NOT_SUPPORTED means the device cannot
    // provide one and skips the device; any other error is a test failure.
    CUdevResource sm_resource;
    CUresult res = cuDeviceGetDevResource(dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);
    if (res == CUDA_ERROR_NOT_SUPPORTED)
    {
      fprintf(stderr, "test_device_from_green_context_stream: device %d skipped (no SM resource)\n", d);
      continue;
    }
    cuda_try(res);

    CUdevResource sm_group;
    CUdevResource sm_remaining;
    unsigned int ngroups = 1;
    cuda_try(cuDevSmResourceSplitByCount(&sm_group, &ngroups, &sm_resource, &sm_remaining, 0, 1));
    CUdevResourceDesc desc;
    cuda_try(cuDevResourceGenerateDesc(&desc, &sm_group, 1));

    CUgreenCtx green_ctx;
    res = cuGreenCtxCreate(&green_ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
    if (res == CUDA_ERROR_NOT_SUPPORTED)
    {
      fprintf(stderr, "test_device_from_green_context_stream: device %d skipped (no green context)\n", d);
      continue;
    }
    cuda_try(res);

    CUstream green_stream;
    cuda_try(cuGreenCtxStreamCreate(&green_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0));
    const auto stream = cudaStream_t(green_stream);

    cuda_try(cudaSetDevice(0));
    _CCCL_ASSERT(get_device_from_stream(stream) == d, "green-context stream must report its device");

    cuda_try(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    _CCCL_ASSERT(get_device_from_stream(stream) == d, "green-context stream device under capture");
    cudaStreamCaptureStatus status;
    cuda_try(cudaStreamIsCapturing(stream, &status));
    _CCCL_ASSERT(status == cudaStreamCaptureStatusActive, "device query must not invalidate the capture");
    cudaGraph_t graph;
    cuda_try(cudaStreamEndCapture(stream, &graph));
    cuda_try(cudaGraphDestroy(graph));

    cuda_try(cuStreamDestroy(green_stream));
    cuda_try(cuGreenCtxDestroy(green_ctx));
  }

  fprintf(stderr, "test_device_from_green_context_stream: PASSED\n");
#else
  (void) ndevs;
  fprintf(stderr, "test_device_from_green_context_stream: skipped (CTK < 12.5)\n");
#endif
}

int main()
{
  int ndevs;
  cuda_try(cudaGetDeviceCount(&ndevs));

  test_basic_pick_stream();
  test_pick_stream_computation_hint();
  test_pick_stream_specific_device(ndevs);
  test_device_from_capturing_stream(ndevs);
  test_device_from_green_context_stream(ndevs);
  test_launch_kernel_on_picked_stream();
  test_round_robin_streams();
  test_two_handles_isolation();
  test_reset_survives_with_fresh_registry();
}
