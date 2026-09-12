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
 * @brief The must-not-capture set: operations that synchronize with the host,
 *        allocate containers or transfer host data REFUSE (throw) under an
 *        active CUDA stream capture, loudly and safely — the ongoing capture
 *        is left VALID (not invalidated, not wedged) and every refused
 *        operation works eagerly after the capture ends. Also pins the
 *        benign cases: shard adoption (host-only bookkeeping) and
 *        `place_group` stream materialization record nothing into the graph.
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct is_even_op
{
  __host__ __device__ bool operator()(long long x) const
  {
    return (x % 2) == 0;
  }
};

bool capture_active(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_safe_call(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatusActive;
}

// Run op during an active capture (forked to data's shard streams), expect a
// std::runtime_error, and require the capture to remain ACTIVE.
template <typename Fn>
void expect_refusal_keeps_capture(cudaStream_t origin, Fn&& op)
{
  bool threw = false;
  try
  {
    op();
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));
}

void test_refusals(place_group& group)
{
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL);
  ::std::vector<long long> host(n, 1);

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);

  // Allocation (owning and contiguous)
  expect_refusal_keeps_capture(origin, [&] {
    auto other = sharded_array<long long>::allocate(group, n);
    (void) other;
  });
  expect_refusal_keeps_capture(origin, [&] {
    auto other = sharded_array<long long>::allocate_contiguous(group, n);
    (void) other;
  });

  // Host transfers
  expect_refusal_keeps_capture(origin, [&] {
    data.copy_from_host(host.data());
  });
  expect_refusal_keeps_capture(origin, [&] {
    data.copy_to_host(host.data());
  });

  // Synchronization entry points
  expect_refusal_keeps_capture(origin, [&] {
    data.sync();
  });
  expect_refusal_keeps_capture(origin, [&] {
    data.sync(0); // per-shard member refuses the same way
  });
  expect_refusal_keeps_capture(origin, [&] {
    group.sync();
  });

  // Other synchronous collectives (host-side combine or size write-back)
  expect_refusal_keeps_capture(origin, [&] {
    (void) count(data, 7LL);
  });
  expect_refusal_keeps_capture(origin, [&] {
    (void) histogram_even(data, 8, 0LL, static_cast<long long>(n));
  });
  expect_refusal_keeps_capture(origin, [&] {
    (void) copy_if(data, is_even_op{});
  });
  expect_refusal_keeps_capture(origin, [&] {
    (void) unique(data);
  });

  // A synchronous elementwise call refuses at ENTRY, before any kernel is
  // recorded — still without invalidating the capture
  expect_refusal_keeps_capture(origin, [&] {
    fill(data, 3LL); // synchronous form
  });

  // Abandon this capture (it holds the blocking fill's kernels)
  cudaGraph_t graph = nullptr;
  data.join_into(origin);
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  cuda_safe_call(cudaGraphDestroy(graph)); // never instantiated

  // Everything refused above works eagerly after the capture
  auto other = sharded_array<long long>::allocate(group, n);
  (void) other;
  data.copy_from_host(host.data());
  data.copy_to_host(host.data());
  EXPECT(count(data, 1LL) == n); // copy_from_host wrote all ones
  data.sync();
  group.sync();

  cuda_safe_call(cudaStreamDestroy(origin));
}

// The lane-ordered capture guard: an asynchronous call whose call stream is
// capturing while the shard environments' streams are NOT must refuse at
// entry (the work would silently escape the graph), leaving the capture
// valid; forking the lanes first makes the same call legal.
void test_lane_ordered_capture_guard(place_group& group)
{
  const size_t n = 4096;
  auto data      = sharded_array<float>::allocate(group, n);
  fill(data, 1.0f);
  data.sync();

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));

  const auto cprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  const auto ce    = ::cuda::std::execution::env{cprop};
  bool threw       = false;
  try
  {
    fill(data, default_envs(data), 2.0f, ce); // lanes not forked: refused
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  // Forked lanes: the same call is legal and records.
  data.fork_from(origin);
  fill(data, default_envs(data), 2.0f, ce);
  data.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));
  ::std::vector<float> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2.0f);
  }
  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}

// Adoption is host-only bookkeeping: it is permitted during capture, and the
// adopted view is usable by captured elementwise work.
void test_adoption_is_benign(place_group& group)
{
  const size_t n = 4096;
  auto owner     = sharded_array<float>::allocate(group, n);
  fill(owner, 1.0f);

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  owner.fork_from(origin);

  auto view = owner.slice(0, n); // adoption path (non-owning view)
  EXPECT(view.is_view());
  EXPECT(capture_active(origin));
  const auto view_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  fill(view, default_envs(view), 2.0f, ::cuda::std::execution::env{view_prop}); // captured through the view (async
                                                                                // form)

  owner.join_into(origin);
  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  ::std::vector<float> host(n);
  owner.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2.0f);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}

// place_group stream materialization during capture is benign: nothing is
// recorded into the graph, the capture stays valid, and the group works
// normally afterwards. (Construct groups and materialize streams BEFORE
// capture anyway: first-touch process initialization is not similarly
// guaranteed.)
void test_group_materialization_records_nothing()
{
  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  auto group = place_group{make_locality_domain_grid()};

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  cudaStream_t s0 = group.get_stream(0); // lazy pool materialization
  EXPECT(s0 != nullptr);
  EXPECT(capture_active(origin));

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  size_t num_nodes = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  EXPECT(num_nodes == 0);
  cuda_safe_call(cudaGraphDestroy(graph));

  // The group is fully usable after
  auto data = sharded_array<int>::allocate(group, 1000);
  fill(data, 5);
  ::std::vector<int> host(1000);
  data.copy_to_host(host.data());
  for (int v : host)
  {
    EXPECT(v == 5);
  }

  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_refusals(group);
  test_lane_ordered_capture_guard(group);
  test_adoption_is_benign(group);
  test_group_materialization_records_nothing();

  return 0;
}
