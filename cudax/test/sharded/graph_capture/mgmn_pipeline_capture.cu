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
 * @brief A back-to-back MGMN pipeline over one sharded array — transform,
 *        reduce (into device scalars), inclusive_scan, exclusive_scan,
 *        transform — with no host synchronization between the calls:
 *        (a) eagerly, lane-ordered, with a single final join, and (b)
 *        captured into ONE CUDA graph through the fork_from/join_into
 *        pattern, instantiated, and replayed with inputs mutated between
 *        launches. The existing `sharded::inclusive_sum` keeps refusing
 *        under capture (its host prefix), while the MGMN path captures.
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct twice_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return 2 * x;
  }
};

struct plus_one_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return x + 1;
  }
};

constexpr long long exclusive_init = 3;

// The pipeline, enqueued lane-ordered on the array's environments: each call
// orders after the previous one per lane by stream order; the cross-lane
// steps inside reduce and the scans are event edges. No host sync anywhere.
template <class CallEnv>
void enqueue_pipeline(sharded_array<long long>& data, ::std::vector<long long*>& lane_outs, const CallEnv& call_env)
{
  const auto envs = default_envs(data);
  transform(data, envs, twice_op{}, call_env);
  mgmn::reduce_into_lanes(data, envs, lane_outs, ::cuda::std::plus<long long>{}, 0LL, 0LL, call_env);
  mgmn::inclusive_sum(data, envs, call_env);
  mgmn::exclusive_sum(data, envs, exclusive_init, call_env);
  transform(data, envs, plus_one_op{}, call_env);
}

// Host reference of the pipeline: returns the array and the reduce value
long long reference(::std::vector<long long>& v)
{
  long long total = 0;
  for (auto& x : v)
  {
    x *= 2;
    total += x;
  }
  long long running = 0;
  for (auto& x : v)
  {
    running += x;
    x = running;
  }
  running = exclusive_init;
  for (auto& x : v)
  {
    const long long cur = x;
    x                   = running;
    running += cur;
  }
  for (auto& x : v)
  {
    x += 1;
  }
  return total;
}

void fill_input(::std::vector<long long>& input, int round)
{
  for (size_t i = 0; i < input.size(); i++)
  {
    input[i] = static_cast<long long>((i * 31 + static_cast<size_t>(round) * 17) % 101) - 50;
  }
}

void check(const sharded_array<long long>& data,
           const long long* d_lanes,
           size_t P,
           const ::std::vector<long long>& expected,
           long long expected_total)
{
  ::std::vector<long long> host(data.size());
  data.copy_to_host(host.data());
  for (size_t i = 0; i < host.size(); i++)
  {
    EXPECT(host[i] == expected[i]);
  }
  ::std::vector<long long> lanes(P);
  cuda_safe_call(cudaMemcpy(lanes.data(), d_lanes, P * sizeof(long long), cudaMemcpyDefault));
  for (size_t g = 0; g < P; g++)
  {
    EXPECT(lanes[g] == expected_total);
  }
}

bool capture_active(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_safe_call(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatusActive;
}

void test_pipeline(place_group& group)
{
  const size_t n = (1 << 20) + 37;
  auto data      = sharded_array<long long>::allocate(group, n);
  const size_t P = data.num_shards();

  long long* d_lanes = nullptr;
  cuda_safe_call(cudaMalloc(&d_lanes, P * sizeof(long long)));
  ::std::vector<long long*> lane_outs;
  for (size_t g = 0; g < P; g++)
  {
    lane_outs.push_back(d_lanes + g);
  }

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  const auto cprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  const auto ce    = ::cuda::std::execution::env{cprop};

  ::std::vector<long long> input(n);
  ::std::vector<long long> expected(n);

  // (a) Eager, asynchronous: fork the lanes from the origin once, enqueue the
  // whole chain, join once, synchronize the origin only.
  fill_input(input, 0);
  data.copy_from_host(input.data());
  expected                = input;
  const long long total_a = reference(expected);
  data.fork_from(origin);
  enqueue_pipeline(data, lane_outs, ce);
  data.join_into(origin);
  cuda_safe_call(cudaStreamSynchronize(origin));
  check(data, d_lanes, P, expected, total_a);

  // (b) Captured into one graph
  fill_input(input, 1);
  data.copy_from_host(input.data());

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);

  // The existing synchronous sharded scan refuses under capture (unchanged
  // behavior), leaving the capture active...
  bool threw = false;
  try
  {
    inclusive_sum(data);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  // ...while the MGMN path captures.
  enqueue_pipeline(data, lane_outs, ce);
  data.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  size_t num_nodes = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  EXPECT(num_nodes > 0);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));

  // Replay twice, inputs mutated between launches (outside the graph)
  for (int round = 1; round <= 2; round++)
  {
    fill_input(input, round);
    data.copy_from_host(input.data());
    expected              = input;
    const long long total = reference(expected);
    cuda_safe_call(cudaGraphLaunch(exec, origin));
    cuda_safe_call(cudaStreamSynchronize(origin));
    check(data, d_lanes, P, expected, total);
  }

  // Eager work is unaffected afterwards (communicator state not wedged)
  fill_input(input, 5);
  data.copy_from_host(input.data());
  expected              = input;
  const long long total = reference(expected);
  data.fork_from(origin);
  enqueue_pipeline(data, lane_outs, ce);
  data.join_into(origin);
  cuda_safe_call(cudaStreamSynchronize(origin));
  check(data, d_lanes, P, expected, total);

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
  cuda_safe_call(cudaFree(d_lanes));
}

// The bracketed composition: sealed against the call stream per call, so a
// plain synchronize of the call stream is the join — also under capture.
void test_bracketed(place_group& group)
{
  const size_t n = 4099;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL);

  cudaStream_t call;
  cuda_safe_call(cudaStreamCreate(&call));
  const auto sprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{call}};
  const auto bprop = ::cuda::std::execution::prop{get_composition_t{}, composition::bracketed};
  const auto ce    = ::cuda::std::execution::env{sprop, bprop};

  mgmn::inclusive_sum(data, ce);
  mgmn::exclusive_sum(data, 1LL, ce);
  cuda_safe_call(cudaStreamSynchronize(call)); // the bracket's join
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  long long running = 1;
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == running);
    running += static_cast<long long>(i) * (static_cast<long long>(i) + 1) / 2;
  }
  cuda_safe_call(cudaStreamDestroy(call));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_pipeline(group);
  test_bracketed(group);

  return 0;
}
