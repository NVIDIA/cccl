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
 * @brief Work submitted into a place's stream executes in the stream's
 *        context with the stream's SM confinement, independent of the
 *        calling thread's current context: `stream_scope` (device currency
 *        derived from the stream) is a sufficient replacement for exec-place
 *        activation on the generic algorithm path.
 *
 * Arms, per locality domain i, launching into the domain's pool stream:
 *   A  activated   : exec_place_scope around the launch (reference behavior)
 *   B  stream_scope: no activation, device currency only
 *   C  adversarial : the OTHER domain's place activated, launch into i's
 *                    stream (kernels must follow the stream, not the current
 *                    context)
 *   D  thrust      : par_nosync.on(stream), no activation
 *   E  events      : event created without activation, recorded/waited on
 *                    the pool streams
 *
 * Gate: arms B, C, D reproduce arm A's per-domain SM set exactly, and every
 * arm computes correct results. Cross-domain SM-set disjointness is
 * reported, not gated (it depends on the SM split method and backend).
 */

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place_scope;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
__global__ void smid_probe(unsigned* smids, float* data, int n)
{
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0)
  {
    smids[blockIdx.x] = s;
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    data[i] = data[i] * 2.0f + 1.0f;
  }
}

constexpr int NB  = 4096;
constexpr int TPB = 128;
constexpr int N   = NB * TPB;

std::set<unsigned> collect_smids(const unsigned* d_smids, cudaStream_t s)
{
  std::vector<unsigned> h(NB);
  cuda_safe_call(cudaMemcpyAsync(h.data(), d_smids, NB * sizeof(unsigned), cudaMemcpyDeviceToHost, s));
  cuda_safe_call(cudaStreamSynchronize(s));
  return std::set<unsigned>(h.begin(), h.end());
}

bool check_data(const float* d_data, cudaStream_t s, float expect)
{
  std::vector<float> h(8);
  cuda_safe_call(cudaMemcpyAsync(h.data(), d_data, 8 * sizeof(float), cudaMemcpyDeviceToHost, s));
  cuda_safe_call(cudaStreamSynchronize(s));
  return std::all_of(h.begin(), h.end(), [&](float v) {
    return v == expect;
  });
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group     = place_group{make_locality_domain_grid()};
  const size_t P = group.size();

  std::vector<unsigned*> d_smids(P);
  std::vector<float*> d_data(P);
  std::vector<cudaStream_t> streams(P);
  for (size_t i = 0; i < P; i++)
  {
    cuda_safe_call(cudaMalloc(&d_smids[i], NB * sizeof(unsigned)));
    cuda_safe_call(cudaMalloc(&d_data[i], N * sizeof(float)));
    cuda_safe_call(cudaMemset(d_data[i], 0, N * sizeof(float)));
    streams[i] = group.get_stream(i);
  }

  auto launch = [&](size_t i) {
    smid_probe<<<NB, TPB, 0, streams[i]>>>(d_smids[i], d_data[i], N);
    cuda_safe_call(cudaGetLastError());
  };

  std::vector<std::set<unsigned>> smA(P), smB(P), smC(P), smD(P);

  // Arm A: reference — exec-place activation around the launch
  for (size_t i = 0; i < P; i++)
  {
    exec_place_scope scope(group.place(i));
    launch(i);
    smA[i] = collect_smids(d_smids[i], streams[i]);
    EXPECT(check_data(d_data[i], streams[i], 1.0f));
  }

  // Arm B: stream_scope only — no context activation anywhere
  for (size_t i = 0; i < P; i++)
  {
    stream_scope scope(streams[i]);
    launch(i);
    smB[i] = collect_smids(d_smids[i], streams[i]);
    EXPECT(check_data(d_data[i], streams[i], 3.0f));
    EXPECT(smB[i] == smA[i]);
  }

  // Arm C: adversarial — the WRONG domain's context is current
  if (P >= 2)
  {
    for (size_t i = 0; i < P; i++)
    {
      exec_place_scope wrong(group.place((i + 1) % P));
      launch(i);
      smC[i] = collect_smids(d_smids[i], streams[i]);
      EXPECT(check_data(d_data[i], streams[i], 7.0f));
      EXPECT(smC[i] == smA[i]);
    }
  }

  // Arm D: thrust through the stream, no activation
  for (size_t i = 0; i < P; i++)
  {
    thrust::transform(
      thrust::cuda::par_nosync.on(streams[i]), d_data[i], d_data[i] + N, d_data[i], [] __device__(float v) {
        return v + 1.0f;
      });
    cuda_safe_call(cudaGetLastError());
    stream_scope scope(streams[i]);
    launch(i); // (x + 1) * 2 + 1
    smD[i]             = collect_smids(d_smids[i], streams[i]);
    const float expect = P >= 2 ? 17.0f : 9.0f; // arm C skipped when P == 1
    EXPECT(check_data(d_data[i], streams[i], expect));
    EXPECT(smD[i] == smA[i]);
  }

  // Arm E: event created without activation, used across pool streams
  if (P >= 2)
  {
    cudaEvent_t ev;
    cuda_safe_call(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    cuda_safe_call(cudaEventRecord(ev, streams[0]));
    cuda_safe_call(cudaStreamWaitEvent(streams[1], ev, 0));
    cuda_safe_call(cudaStreamSynchronize(streams[1]));
    cuda_safe_call(cudaEventDestroy(ev));
  }

  // Informational: cross-domain disjointness (depends on split method)
  if (P >= 2)
  {
    const bool disjoint = std::none_of(smA[0].begin(), smA[0].end(), [&](unsigned v) {
      return smA[1].count(v) != 0;
    });
    std::printf("stream_scope: %zu domains, dom0 %zu SMs, dom1 %zu SMs, disjoint=%d\n",
                P,
                smA[0].size(),
                smA[1].size(),
                static_cast<int>(disjoint));
  }

  for (size_t i = 0; i < P; i++)
  {
    cuda_safe_call(cudaFree(d_smids[i]));
    cuda_safe_call(cudaFree(d_data[i]));
  }

  return 0;
}
