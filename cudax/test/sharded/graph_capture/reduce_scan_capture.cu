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
 * @brief Reduce and scan under CUDA graph capture. The synchronous forms
 *        (the value-returning `reduce` / `sum`, the no-stream scans) REFUSE
 *        cleanly (throw) when invoked during capture — without invalidating
 *        the ongoing capture — and keep working eagerly afterwards. The
 *        stream-bearing forms (`reduce_into`, `reduce_into_lanes`, the scans
 *        with a call environment) are pure stream work on their MGMN engines:
 *        they CAPTURE into the same graph, which instantiates and replays
 *        correctly with inputs mutated between launches.
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

constexpr long long exclusive_init = 5;
constexpr long long lanes_init     = 100;

bool capture_active(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_safe_call(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatusActive;
}

void fill_input(::std::vector<long long>& input, int round)
{
  for (size_t i = 0; i < input.size(); i++)
  {
    input[i] = static_cast<long long>(i) + round;
  }
}

// Host reference of the captured chain; returns the reduce total
long long reference(::std::vector<long long>& v)
{
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
  long long total = 0;
  for (const auto x : v)
  {
    total += x;
  }
  for (auto& x : v)
  {
    x += 1;
  }
  return total;
}

void check(const sharded_array<long long>& data,
           const long long* d_total,
           const long long* d_lanes,
           const ::std::vector<long long>& expected,
           long long expected_total)
{
  ::std::vector<long long> host(data.size());
  data.copy_to_host(host.data());
  for (size_t i = 0; i < host.size(); i++)
  {
    EXPECT(host[i] == expected[i]);
  }
  long long total = 0;
  cuda_safe_call(cudaMemcpy(&total, d_total, sizeof(long long), cudaMemcpyDefault));
  EXPECT(total == expected_total);
  ::std::vector<long long> lanes(data.num_shards());
  cuda_safe_call(cudaMemcpy(lanes.data(), d_lanes, lanes.size() * sizeof(long long), cudaMemcpyDefault));
  for (const auto lane : lanes)
  {
    EXPECT(lane == lanes_init + expected_total);
  }
}

void test_reduce_scan_capture(place_group& group)
{
  const size_t n  = 100003;
  auto data       = sharded_array<long long>::allocate(group, n);
  const size_t P  = data.num_shards();
  const auto envs = default_envs(data);
  ::std::vector<long long> input(n);
  ::std::vector<long long> expected(n);
  fill_input(input, 0);
  data.copy_from_host(input.data());

  long long* d_total = nullptr;
  long long* d_lanes = nullptr;
  cuda_safe_call(cudaMalloc(&d_total, sizeof(long long)));
  cuda_safe_call(cudaMalloc(&d_lanes, P * sizeof(long long)));

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  const auto cprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  const auto ce    = ::cuda::std::execution::env{cprop};

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);

  // Every synchronous form refuses with std::runtime_error and leaves the
  // capture ACTIVE
  bool threw = false;
  try
  {
    (void) reduce(data, ::cuda::std::plus<long long>{}, 0LL);
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
    (void) sum(data);
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
    inclusive_sum(data); // no-stream form: synchronous convenience
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
    exclusive_sum(data, exclusive_init); // no-stream form
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(capture_active(origin));

  // The stream-bearing forms record into the capture
  inclusive_sum(data, envs, ce);
  exclusive_sum(data, envs, exclusive_init, ce);
  reduce_into(data, envs, d_total, ::cuda::std::plus<long long>{}, 0LL, ce);
  reduce_into_lanes(data, envs, d_lanes, ::cuda::std::plus<long long>{}, lanes_init);
  transform(data, envs, plus_one_op{}, ce);
  data.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  size_t num_nodes = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  EXPECT(num_nodes > 0);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));

  // The refused calls left no work behind: the first replay computes the
  // chain from the untouched input; further replays from mutated inputs
  for (int round = 0; round <= 2; round++)
  {
    fill_input(input, round);
    data.copy_from_host(input.data());
    expected              = input;
    const long long total = reference(expected);
    cuda_safe_call(cudaGraphLaunch(exec, origin));
    cuda_safe_call(cudaStreamSynchronize(origin));
    check(data, d_total, d_lanes, expected, total);
  }

  // Eager reduce/scan work normally after the capture (state not wedged)
  fill_input(input, 7);
  data.copy_from_host(input.data());
  expected              = input;
  const long long total = reference(expected);
  inclusive_sum(data);
  exclusive_sum(data, exclusive_init);
  EXPECT(sum(data) == total);
  transform(data, plus_one_op{});
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == expected[i]);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
  cuda_safe_call(cudaFree(d_total));
  cuda_safe_call(cudaFree(d_lanes));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_reduce_scan_capture(group);

  return 0;
}
