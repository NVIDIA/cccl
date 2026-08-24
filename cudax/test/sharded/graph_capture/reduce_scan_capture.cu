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
 *
 * @brief Reduce and scan under CUDA graph capture: both have a host-side
 *        combine phase and per-call temporaries, so the shipped contract is
 *        to REFUSE cleanly (throw) when invoked during capture — without
 *        invalidating the ongoing capture, which stays usable for supported
 *        (elementwise) work — and to keep working eagerly afterwards.
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
struct plus_one_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return x + 1;
  }
};

bool capture_active(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_safe_call(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatusActive;
}

void test_reduce_scan_refuse_under_capture(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(group, data, 0LL);

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);

  // Every refusal must throw std::runtime_error and leave the capture ACTIVE
  bool threw = false;
  try
  {
    (void) reduce(group, data, ::cuda::std::plus<long long>{}, 0LL);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  threw = false;
  try
  {
    (void) sum(group, data);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  threw = false;
  try
  {
    inclusive_scan(group, data);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  threw = false;
  try
  {
    exclusive_scan(group, data, 5LL);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  // The capture is still usable for supported work: record an elementwise op
  transform(group, data, plus_one_op{}, /*blocking=*/false);
  data.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  // The refused calls left no partial work behind: data is exactly iota + 1
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }

  // Eager reduce/scan work normally after the capture (state not wedged)
  const long long total = sum(group, data);
  // sum of (i+1) for i in [0, n) = n*(n+1)/2
  EXPECT(total == static_cast<long long>(n) * static_cast<long long>(n + 1) / 2);

  inclusive_scan(group, data);
  data.copy_to_host(host.data());
  long long running = 0;
  for (size_t i = 0; i < n; i++)
  {
    running += static_cast<long long>(i) + 1;
    EXPECT(host[i] == running);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_reduce_scan_refuse_under_capture(group);

  return 0;
}
