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
 * @brief A two-field iterative pipeline under the lane-ordered composition
 *        contract: independent per-field work overlaps by default, the one
 *        genuine coupling per iteration is a declared edge, and the
 *        convergence check synchronizes with the RESULT's timeline — never
 *        with the fields' lanes.
 *
 * The contract in one rule: an asynchronous call enqueues each shard's work
 * on its environment's stream and touches nothing else. Consecutive calls on
 * the same environments are ordered per lane by stream order; everything
 * else is said explicitly:
 *
 *  - Two fields x and y live on DISTINCT lanes (`allocate(..., lane_id)`),
 *    so their per-iteration map chains run concurrently — inspect a timeline
 *    capture (e.g. nsys) to see the two fields' kernels overlap.
 *  - Once per iteration, y consumes a scalar reduced from x. The reduction
 *    delivers its aggregate on a dedicated coupling stream (`reduce_into`:
 *    the result rides the OUTPUT's timeline), and y's lanes wait for that
 *    one timeline with `lane_wait` — a single declared cross-field edge,
 *    not a global barrier.
 *  - The convergence residual is reduced into a pinned host slot on its own
 *    stream; the host inspects it by synchronizing THAT stream only, while
 *    the lanes are already running later iterations (software pipelining
 *    across iterations falls out of lane order).
 */

#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct decay_op // x <- a*x + b (kept uniform so the reference is closed-form)
{
  __device__ float operator()(float v) const
  {
    return 0.5f * v + 1.0f;
  }
};

struct couple_op // y <- y + s / n, s read from the reduction's device slot
{
  const float* s_slot;
  float inv_n;
  __device__ float operator()(float v) const
  {
    return v + (*s_slot) * inv_n;
  }
};
} // namespace

int main()
{
  cuda_safe_call(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};
  std::printf("place_group with %zu place(s)\n", group.size());

  const size_t n = size_t{1} << 22;
  auto x         = sharded_array<float>::allocate(group, n, /*lane_id*/ 0);
  auto y         = sharded_array<float>::allocate(group, n, /*lane_id*/ 1);
  auto envs_x    = default_envs(x);
  auto envs_y    = default_envs(y);

  fill(x, 4.0f);
  fill(y, 0.0f);

  // The coupling stream (x's reduction delivers here) and the residual
  // stream (the convergence result delivers here), plus their call envs.
  cudaStream_t cx, cr;
  cuda_safe_call(cudaStreamCreate(&cx));
  cuda_safe_call(cudaStreamCreate(&cr));
  const auto prop_cx = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cx}};
  const auto prop_cr = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cr}};
  const auto ce_x    = ::cuda::std::execution::env{prop_cx};
  const auto ce_r    = ::cuda::std::execution::env{prop_cr};

  // Device slot for the coupling scalar; pinned slot for the residual.
  float* d_s = nullptr;
  cuda_safe_call(cudaMalloc(&d_s, sizeof(float)));
  float* h_res = nullptr;
  cuda_safe_call(cudaMallocHost(&h_res, sizeof(float)));

  // A one-env range around the coupling stream: foreign environments model
  // the concept too, so the verbs work on them unchanged.
  const auto cx_env = ::cuda::std::execution::env{prop_cx};
  const ::std::vector cx_range{cx_env};

  // Host-side closed-form reference (both fields stay uniform-valued).
  double ref_x = 4.0, ref_y = 0.0;

  const int iters = 20;
  for (int k = 0; k < iters; k++)
  {
    // Per-field map chains: lane-ordered, independent across the two fields
    // (distinct lanes) — these overlap.
    transform(x, envs_x, decay_op{}, ce_x);
    transform(x, envs_x, decay_op{}, ce_x);
    transform(y, envs_y, decay_op{}, ce_r);

    // The coupling: reduce x into the device slot on the coupling stream
    // (the aggregate rides the OUTPUT's timeline) ...
    reduce_into(x, envs_x, d_s, ::cuda::std::plus<float>{}, 0.0f, ce_x);
    // ... and y's lanes wait for that one timeline: the declared edge.
    for (size_t i = 0; i < envs_y.size(); i++)
    {
      lane_wait(envs_y, i, cx_range, {0});
    }
    transform(y, envs_y, couple_op{d_s, 1.0f / static_cast<float>(n)}, ce_r);
    // Slot reuse needs the reverse edge: the NEXT iteration's reduction
    // overwrites d_s on the coupling stream, so that stream must first wait
    // for y's read of this iteration's value (or double-buffer the slot).
    barrier(envs_y, ::cuda::stream_ref{cx});

    // Convergence residual on its own stream (sum(y) here, as a stand-in
    // for a real residual): the host will await THIS timeline only.
    reduce_into(y, envs_y, h_res, ::cuda::std::plus<float>{}, 0.0f, ce_r);

    // Reference: x and y remain uniform; two decays, then the coupling.
    ref_x = 0.5 * (0.5 * ref_x + 1.0) + 1.0;
    ref_y = (0.5 * ref_y + 1.0) + ref_x; // + sum(x)/n == + x's uniform value
  }

  // Result-attached synchronization: await the residual stream — not the
  // lanes, not a global barrier.
  cuda_safe_call(cudaStreamSynchronize(cr));
  const double expected_res = ref_y * static_cast<double>(n);
  const double got          = static_cast<double>(*h_res);
  std::printf("residual after %d iterations: %.6e (expected %.6e)\n", iters, got, expected_res);

  // Drain everything before verifying the fields themselves.
  barrier(envs_x);
  barrier(envs_y);
  ::std::vector<float> host(n);
  y.copy_to_host(host.data());
  bool ok = std::abs(got - expected_res) <= 1e-6 * std::abs(expected_res);
  for (size_t i = 0; i < n; i++)
  {
    ok = ok && (std::abs(static_cast<double>(host[i]) - ref_y) <= 1e-5 * std::abs(ref_y));
  }

  cuda_safe_call(cudaFree(d_s));
  cuda_safe_call(cudaFreeHost(h_res));
  cuda_safe_call(cudaStreamDestroy(cx));
  cuda_safe_call(cudaStreamDestroy(cr));

  if (!ok)
  {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
