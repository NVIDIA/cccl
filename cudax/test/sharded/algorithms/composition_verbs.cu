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
 * @brief The composition verbs: host barrier (with forbid/capture
 *        refusals), stream barrier (event-based, capture-legal), lane_wait
 *        (same-range and cross-range forms, dependency correctness, capture
 *        legality), lane_sync, and the per-call composition property
 *        (default = lane_ordered, bracketed readable off a call env).
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
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

struct copy_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return x;
  }
};

bool capture_active(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_safe_call(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatusActive;
}

void test_property()
{
  EXPECT(query_composition(default_call_env{}) == composition::lane_ordered);
  const auto cprop = ::cuda::std::execution::prop{get_composition_t{}, composition::bracketed};
  const auto ce    = ::cuda::std::execution::env{cprop};
  EXPECT(query_composition(ce) == composition::bracketed);
}

void test_host_barrier(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);
  auto envs      = default_envs(data);
  iota(data, 0LL);

  // Work on the lanes, then the explicit global join; results must be
  // host-visible afterwards.
  transform(data, envs, plus_one_op{});
  barrier(envs);
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }

  // forbid: refused before any CUDA call.
  bool threw             = false;
  const auto forbid_prop = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
  try
  {
    barrier(envs, ::cuda::std::execution::env{forbid_prop});
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);

  // capture: refused, capture stays valid.
  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  threw = false;
  try
  {
    barrier(envs);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));
  cudaGraph_t g = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &g));
  cuda_safe_call(cudaGraphDestroy(g));
  cuda_safe_call(cudaStreamDestroy(origin));
}

void test_stream_barrier(place_group& group)
{
  const size_t n = 200003;
  auto data      = sharded_array<long long>::allocate(group, n);
  auto envs      = default_envs(data);
  iota(data, 5LL);

  // Join all lanes into one fresh stream; syncing only that stream must
  // suffice to observe the lanes' results.
  transform(data, envs, plus_one_op{});
  cudaStream_t s;
  cuda_safe_call(cudaStreamCreate(&s));
  barrier(envs, ::cuda::stream_ref{s});
  cuda_safe_call(cudaStreamSynchronize(s));
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 6);
  }
  cuda_safe_call(cudaStreamDestroy(s));
}

void test_lane_wait(place_group& group)
{
  if (group.size() < 2)
  {
    return; // needs two lanes
  }
  const size_t per = 100003;
  const ::std::vector<size_t> sizes(group.size(), per); // one shard per place
  // x on one lane set, y on another (distinct lane_ids = independent lanes).
  auto x      = sharded_array<long long>::allocate(group, sizes, 0);
  auto y      = sharded_array<long long>::allocate(group, sizes, 1);
  auto envs_x = default_envs(x);
  auto envs_y = default_envs(y);

  iota(x, 0LL);
  fill(y, -1LL);

  // Enqueue work on x's lanes, declare the cross-field dependency, then
  // consume x from y's lanes: correctness is guaranteed by the edges, not
  // by timing.
  transform(x, envs_x, plus_one_op{});
  for (size_t i = 0; i < envs_y.size(); i++)
  {
    lane_wait(envs_y, i, envs_x, {i});
  }
  zip_transform(y, envs_y, copy_op{}, default_call_env{}, x);
  barrier(envs_y);

  ::std::vector<long long> host(group.size() * per);
  y.copy_to_host(host.data());
  for (size_t i = 0; i < host.size(); i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }

  // Same-range form + out-of-range refusals.
  lane_wait(envs_x, 0, {1});
  bool threw = false;
  try
  {
    lane_wait(envs_x, 99, {0});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    lane_wait(envs_x, 0, {99});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_lane_wait_capture_legal(place_group& group)
{
  const size_t n = 4096;
  auto data      = sharded_array<long long>::allocate(group, n);
  auto envs      = default_envs(data);
  iota(data, 0LL);
  data.sync();

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);

  const auto cprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  transform(data, envs, plus_one_op{}, ::cuda::std::execution::env{cprop});

  // Event edges under capture: lane 0 waits for every other lane, and the
  // stream barrier joins the lanes back into the origin.
  if (envs.size() > 1)
  {
    lane_wait(envs, 0, {1});
  }
  barrier(envs, ::cuda::stream_ref{origin});

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }
  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}

void test_lane_sync(place_group& group)
{
  const size_t n = 50021;
  auto data      = sharded_array<long long>::allocate(group, n);
  auto envs      = default_envs(data);
  fill(data, 3LL);
  lane_sync(envs, 0);

  bool threw             = false;
  const auto forbid_prop = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
  try
  {
    lane_sync(envs, 0, ::cuda::std::execution::env{forbid_prop});
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    lane_sync(envs, 99);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}
} // namespace

int main()
{
  cuda_safe_call(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_property();
  test_host_barrier(group);
  test_stream_barrier(group);
  test_lane_wait(group);
  test_lane_wait_capture_legal(group);
  test_lane_sync(group);

  printf("composition_verbs: all tests passed\n");
  return 0;
}
