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
 * @brief `reduce_into`, the asynchronous reduce: device and pinned output
 *        locations, bitwise parity with the synchronous form's fold,
 *        legality under sync_policy::forbid, and — the headline — capture
 *        into a replayable CUDA graph together with a transform (the
 *        solver-loop shape: iterate + reduce a residual to a pinned slot,
 *        entirely graph-resident).
 */

#include <cuda/experimental/sharded.cuh>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct plus2
{
  __host__ __device__ double operator()(double v) const
  {
    return v + 2.0;
  }
};
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  auto group     = place_group{make_locality_domain_grid()};
  const size_t n = 500000;
  auto a         = sharded_array<double>::allocate(group, n);
  fill(a, 1.0);
  auto envs = default_envs(a);

  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));
  const auto sp = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}};
  const auto ce = ::cuda::std::execution::env{sp};

  // Device output location
  double* d_out;
  cuda_safe_call(cudaMalloc(&d_out, sizeof(double)));
  reduce_into(a, envs, d_out, ::cuda::std::plus<double>{}, 0.0, ce);
  double h = 0;
  cuda_safe_call(cudaMemcpyAsync(&h, d_out, sizeof(double), cudaMemcpyDeviceToHost, cs));
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(h == static_cast<double>(n));

  // Pinned output location (self-bound overload), bitwise vs the sync fold
  double* h_out;
  cuda_safe_call(cudaMallocHost(&h_out, sizeof(double)));
  *h_out = -1.0;
  reduce_into(a, h_out, ::cuda::std::plus<double>{}, 0.0, ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == reduce(a, ::cuda::std::plus<double>{}, 0.0));

  // Legal under sync_policy::forbid: no host synchronization inside
  const auto fp  = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
  const auto ce2 = ::cuda::std::execution::env{sp, fp};
  reduce_into(a, d_out, ::cuda::std::plus<double>{}, 0.0, ce2);
  cuda_safe_call(cudaStreamSynchronize(cs));

  // Empty view: writes the init value
  sharded_array<double> empty;
  reduce_into(empty, ::std::vector<shard_env_t>{}, h_out, ::cuda::std::plus<double>{}, 42.0, ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == 42.0);

  // Capture: transform + reduce_into in ONE replayable graph. Lane-ordered
  // capture: fork the lanes from the origin once; reduce_into (a
  // terminator) joins them back into the origin itself.
  fill(a, 1.0);
  cuda_safe_call(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  a.fork_from(cs);
  transform(a, envs, plus2{}, ce); // a += 2, in lane order
  reduce_into(a, envs, h_out, ::cuda::std::plus<double>{}, 0.0, ce);
  cudaGraph_t graph;
  cuda_safe_call(cudaStreamEndCapture(cs, &graph));
  cudaGraphExec_t exec;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  cuda_safe_call(cudaGraphLaunch(exec, cs)); // a: 1 -> 3
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == 3.0 * n);

  cuda_safe_call(cudaGraphLaunch(exec, cs)); // replay, a: 3 -> 5
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == 5.0 * n);

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaFree(d_out));
  cuda_safe_call(cudaFreeHost(h_out));
  cuda_safe_call(cudaStreamDestroy(cs));

  return 0;
}
