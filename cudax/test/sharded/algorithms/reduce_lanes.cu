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
 * @brief The reduce init contract and the lane-resident reduce.
 *
 * Init applied exactly once (`result = init (+) fold(all elements)`) across
 * the three delivery forms — `reduce`, `reduce_into`, `reduce_into_lanes` —
 * on 2 shards (locality-domain group) and on 3 explicitly placed shards, with
 * plus and maximum, with empty shards at the front, in the middle, and
 * everywhere; `reduce_into_lanes` delivering the aggregate on EVERY lane,
 * back-to-back (slot lifetime), and under CUDA graph capture.
 */

#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::data_place;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
// Reference for `plus` over n ones: n + init
double ref_plus_ones(size_t n, double init)
{
  return static_cast<double>(n) + init;
}

//! Check `reduce`, `reduce_into` and `reduce_into_lanes` on @p a against a
//! host-computed reference, for one (op, init) pair.
template <class Op>
void check_all_forms(const sharded_array<double>& a, Op op, double init, double expected, cudaStream_t cs)
{
  auto envs      = default_envs(a);
  const size_t P = a.num_shards();
  const auto sp  = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}};
  const auto ce  = ::cuda::std::execution::env{sp};

  // 1. synchronous
  EXPECT(reduce(a, envs, op, init) == expected);

  // 2. call-stream terminator
  double* h_out;
  cuda_safe_call(cudaMallocHost(&h_out, sizeof(double)));
  *h_out = -12345.0;
  reduce_into(a, envs, h_out, op, init, ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == expected);

  // 3. lane-resident: every lane's slot holds the aggregate, twice
  // back-to-back (a lifetime bug on the partial slots would show here)
  double* h_lanes;
  cuda_safe_call(cudaMallocHost(&h_lanes, (P > 0 ? P : 1) * sizeof(double)));
  for (int rep = 0; rep < 2; ++rep)
  {
    for (size_t g = 0; g < P; ++g)
    {
      h_lanes[g] = -12345.0 - static_cast<double>(rep);
    }
    reduce_into_lanes(a, envs, h_lanes, op, init);
    reduce_into_lanes(a, envs, h_lanes, op, init);
    barrier(envs);
    for (size_t g = 0; g < P; ++g)
    {
      EXPECT(h_lanes[g] == expected);
    }
  }

  cuda_safe_call(cudaFreeHost(h_out));
  cuda_safe_call(cudaFreeHost(h_lanes));
}

//! Ones everywhere: plus with a non-identity init, maximum with an
//! identity-like init and with an init above every element.
void check_ones(const sharded_array<double>& a, size_t n, cudaStream_t cs)
{
  fill(a, 1.0);
  cuda_safe_call(cudaDeviceSynchronize());
  check_all_forms(a, ::cuda::std::plus<double>{}, 1000.0, ref_plus_ones(n, 1000.0), cs);
  check_all_forms(a, ::cuda::std::plus<double>{}, 0.0, ref_plus_ones(n, 0.0), cs);
  // maximum: init below everything -> 1.0 if any element, else init
  check_all_forms(a, ::cuda::maximum<double>{}, -1e300, n > 0 ? 1.0 : -1e300, cs);
  // maximum: init above every element -> the init, exactly once and unchanged
  check_all_forms(a, ::cuda::maximum<double>{}, 5.0, 5.0, cs);
}

//! Three shards on device 0 with their own streams (three lanes), sizes as
//! given (zero allowed: the shard stays, empty).
struct three_lanes
{
  cudaStream_t s[3];
  three_lanes()
  {
    for (auto& st : s)
    {
      cuda_safe_call(cudaStreamCreate(&st));
    }
  }
  ~three_lanes()
  {
    for (auto st : s)
    {
      cuda_safe_call(cudaStreamDestroy(st));
    }
  }
  sharded_array<double> make(size_t n0, size_t n1, size_t n2) const
  {
    return sharded_array<double>::allocate(
      {{n0, data_place::device(0), exec_place::device(0), s[0]},
       {n1, data_place::device(0), exec_place::device(0), s[1]},
       {n2, data_place::device(0), exec_place::device(0), s[2]}});
  }
};

void test_init_two_shards(place_group& group, cudaStream_t cs)
{
  for (const size_t n : {size_t{8388608}, size_t{1000003}})
  {
    auto a = sharded_array<double>::allocate(group, n);
    EXPECT(a.num_shards() == group.size());
    check_ones(a, n, cs);
  }
}

void test_init_three_shards(const three_lanes& lanes, cudaStream_t cs)
{
  for (const size_t n : {size_t{8388608}, size_t{1000003}})
  {
    const size_t n0 = n / 3, n1 = n / 3 + 1, n2 = n - n0 - n1;
    auto a = lanes.make(n0, n1, n2);
    EXPECT(a.num_shards() == 3);
    check_ones(a, n, cs);
  }
}

void test_empty_shards(const three_lanes& lanes, cudaStream_t cs)
{
  // shard 0 empty
  check_ones(lanes.make(0, 1000, 1001), 2001, cs);
  // middle shard empty
  check_ones(lanes.make(777, 0, 1001), 1778, cs);
  // last shard empty
  check_ones(lanes.make(777, 1000, 0), 1777, cs);
  // all shards empty -> init, on every form and every lane
  check_ones(lanes.make(0, 0, 0), 0, cs);
}

//! Lane-resident output consumed BY THE LANES with no further edges: each
//! lane rescales its shard by the global sum, in lane order.
struct plus2
{
  __host__ __device__ double operator()(double v) const
  {
    return v + 2.0;
  }
};

struct scale_by
{
  const double* factor;
  __host__ __device__ double operator()(double v) const
  {
    return v / *factor;
  }
};

void test_lanes_consumed_on_lanes(place_group& group)
{
  const size_t n = 1000003;
  auto a         = sharded_array<double>::allocate(group, n);
  auto envs      = default_envs(a);
  fill(a, 2.0);
  const size_t P = a.num_shards();
  double* d_sums;
  cuda_safe_call(cudaMalloc(&d_sums, P * sizeof(double)));
  reduce_into_lanes(a, envs, d_sums, ::cuda::std::plus<double>{}, 0.0);
  // Per-lane consumer: shard g reads d_sums[g] on lane g — stream order alone
  // orders it after lane g's fold (that is the point of the variant).
  for (size_t g = 0; g < P; ++g)
  {
    const auto& s = a.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    stream_scope scope(s.stream);
    cuda_safe_call(cub::DeviceTransform::Transform(s.data, s.data, s.size, scale_by{d_sums + g}, s.stream));
  }
  barrier(envs);
  ::std::vector<double> host(n);
  a.copy_to_host(host.data());
  const double expect = 2.0 / (2.0 * static_cast<double>(n));
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == expect);
  }
  cuda_safe_call(cudaFree(d_sums));
}

//! Capture: the lanes forked from the origin once, reduce_into_lanes recorded
//! with its cross-lane event edges, lanes joined back; replay twice.
void test_lanes_capture(place_group& group)
{
  const size_t n = 500000;
  auto a         = sharded_array<double>::allocate(group, n);
  auto envs      = default_envs(a);
  const size_t P = a.num_shards();
  fill(a, 1.0);
  cuda_safe_call(cudaDeviceSynchronize());

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  double* h_lanes;
  cuda_safe_call(cudaMallocHost(&h_lanes, P * sizeof(double)));

  const auto sp = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  const auto ce = ::cuda::std::execution::env{sp};

  // Relaxed mode: the locality-domain places resolve the driver's default
  // pool at each stream-ordered allocation, and that resolution queries a
  // pool attribute — an API call the ThreadLocal/Global modes forbid while
  // capturing (error 900, "operation not permitted when stream is
  // capturing"); the allocation itself captures fine. Relaxed lifts the
  // thread-side restriction without changing the graph.
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeRelaxed));
  a.fork_from(origin);
  transform(a, envs, plus2{}, ce); // lane-ordered
  reduce_into_lanes(a, envs, h_lanes, ::cuda::std::plus<double>{}, 1000.0);
  a.join_into(origin);
  cudaGraph_t graph;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  cudaGraphExec_t exec;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  cuda_safe_call(cudaGraphLaunch(exec, origin)); // a: 1 -> 3
  cuda_safe_call(cudaStreamSynchronize(origin));
  for (size_t g = 0; g < P; ++g)
  {
    EXPECT(h_lanes[g] == 3.0 * n + 1000.0);
  }
  cuda_safe_call(cudaGraphLaunch(exec, origin)); // replay, a: 3 -> 5
  cuda_safe_call(cudaStreamSynchronize(origin));
  for (size_t g = 0; g < P; ++g)
  {
    EXPECT(h_lanes[g] == 5.0 * n + 1000.0);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaFreeHost(h_lanes));
  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  auto group = place_group{make_locality_domain_grid(0)};
  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));
  three_lanes lanes;

  test_init_two_shards(group, cs);
  test_init_three_shards(lanes, cs);
  test_empty_shards(lanes, cs);
  test_lanes_consumed_on_lanes(group);
  test_lanes_capture(group);

  cuda_safe_call(cudaStreamDestroy(cs));
  return 0;
}
