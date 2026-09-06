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
 * @brief Generic segmented_reduce: CSR-shaped correctness through the
 *        shifted-alias segments spelling, empty segments, the partition /
 *        alignment refusals, sync-policy refusal, and the asynchronous form
 *        under CUDA graph capture (map-class: capture-legal, balanced
 *        in-capture scratch).
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct sum_op
{
  __host__ __device__ float operator()(float a, float b) const
  {
    return a + b;
  }
};

struct max_op
{
  __host__ __device__ int operator()(int a, int b) const
  {
    return a < b ? b : a;
  }
};

// Ragged segment layout per shard: segment i (0-based within the shard) has
// (i % 4) elements — includes empty segments — with value pattern
// in[k] = (k % 9) + 1 within the shard.
struct ragged_case
{
  ::std::vector<::std::vector<int>> h_off; // per shard, n_seg+1 entries
  ::std::vector<::std::vector<float>> h_in; // per shard values
  ::std::vector<::std::size_t> seg_sizes; // segments per shard
  ::std::vector<::std::size_t> in_sizes; // values per shard
};

ragged_case make_ragged(::std::size_t num_segments, ::std::size_t P)
{
  ragged_case c;
  c.h_off.resize(P);
  c.h_in.resize(P);
  ::std::vector<::std::size_t> per_shard(P, num_segments / P);
  for (::std::size_t g = 0; g < num_segments % P; g++)
  {
    per_shard[g]++;
  }
  for (::std::size_t g = 0; g < P; g++)
  {
    c.h_off[g].push_back(0);
    for (::std::size_t i = 0; i < per_shard[g]; i++)
    {
      for (::std::size_t k = 0; k < i % 4; k++)
      {
        c.h_in[g].push_back(static_cast<float>((c.h_in[g].size() % 9) + 1));
      }
      c.h_off[g].push_back(static_cast<int>(c.h_in[g].size()));
    }
    c.seg_sizes.push_back(per_shard[g]);
    c.in_sizes.push_back(c.h_in[g].size());
  }
  return c;
}

void test_ragged_correctness(place_group& group)
{
  const ::std::size_t P = group.size();
  auto envs             = group.envs(0);
  const auto c          = make_ragged(4097, P);

  // Device offsets buffers (one per shard), shifted-alias views over them.
  ::std::vector<int*> d_off(P);
  ::std::vector<float*> d_in(P);
  ::std::vector<cuda::std::span<const int>> lo(P), hi(P);
  ::std::vector<cuda::std::span<const float>> vin(P);
  for (::std::size_t g = 0; g < P; g++)
  {
    const auto& env = envs[g];
    const auto strm = ::cuda::get_stream(env);
    stream_scope sc(strm.get());
    auto mr  = ::cuda::mr::get_memory_resource(env);
    d_off[g] = static_cast<int*>(mr.allocate(strm, c.h_off[g].size() * sizeof(int), 256));
    d_in[g]  = static_cast<float*>(mr.allocate(strm, (c.h_in[g].empty() ? 1 : c.h_in[g].size()) * sizeof(float), 256));
    cuda_safe_call(cudaMemcpyAsync(
      d_off[g], c.h_off[g].data(), c.h_off[g].size() * sizeof(int), cudaMemcpyHostToDevice, strm.get()));
    cuda_safe_call(
      cudaMemcpyAsync(d_in[g], c.h_in[g].data(), c.h_in[g].size() * sizeof(float), cudaMemcpyHostToDevice, strm.get()));
    cuda_safe_call(cudaStreamSynchronize(strm.get()));
    lo[g]  = {d_off[g], c.seg_sizes[g]};
    hi[g]  = {d_off[g] + 1, c.seg_sizes[g]};
    vin[g] = {d_in[g], c.h_in[g].size()};
  }
  const auto seg_begin = make_sharded_view(lo);
  const auto seg_end   = make_sharded_view(hi);
  const auto in        = make_sharded_view(vin);
  EXPECT(validate(seg_begin));

  auto out = sharded_array<float>::allocate(group, c.seg_sizes, 0);

  // Explicit-envs form.
  segmented_reduce(in, envs, seg_begin, seg_end, out, sum_op{}, 0.0f);

  ::std::vector<float> h_out(out.size());
  out.copy_to_host(h_out.data());
  ::std::size_t v = 0;
  for (::std::size_t g = 0; g < P; g++)
  {
    for (::std::size_t i = 0; i + 1 < c.h_off[g].size(); i++)
    {
      float ref = 0.0f;
      for (int k = c.h_off[g][i]; k < c.h_off[g][i + 1]; k++)
      {
        ref += c.h_in[g][static_cast<::std::size_t>(k)];
      }
      EXPECT(::std::abs(h_out[v] - ref) <= 1e-5f * (1.0f + ::std::abs(ref)));
      // Empty segments got exactly init.
      if (c.h_off[g][i] == c.h_off[g][i + 1])
      {
        EXPECT(h_out[v] == 0.0f);
      }
      v++;
    }
  }

  // Self-bound convenience form (envs derived from the output container).
  fill(out, -1.0f);
  segmented_reduce(in, seg_begin, seg_end, out, sum_op{}, 0.0f);
  ::std::vector<float> h_out2(out.size());
  out.copy_to_host(h_out2.data());
  for (::std::size_t i = 0; i < h_out.size(); i++)
  {
    EXPECT(h_out[i] == h_out2[i]);
  }

  for (::std::size_t g = 0; g < P; g++)
  {
    const auto& env = envs[g];
    const auto strm = ::cuda::get_stream(env);
    auto mr         = ::cuda::mr::get_memory_resource(env);
    mr.deallocate(strm, d_off[g], c.h_off[g].size() * sizeof(int), 256);
    mr.deallocate(strm, d_in[g], (c.h_in[g].empty() ? 1 : c.h_in[g].size()) * sizeof(float), 256);
    cuda_safe_call(cudaStreamSynchronize(strm.get()));
  }
}

void test_refusals(place_group& group)
{
  auto envs             = group.envs(0);
  const ::std::size_t P = group.size();

  ::std::vector<::std::size_t> two_per(P, 2), three_per(P, 3);
  auto out   = sharded_array<int>::allocate(group, two_per, 0);
  auto out3  = sharded_array<int>::allocate(group, three_per, 0);
  auto in    = sharded_array<int>::allocate(group, three_per, 0);
  auto seg_b = sharded_array<int>::allocate(group, two_per, 0);
  auto seg_e = sharded_array<int>::allocate(group, two_per, 0);
  fill(seg_b, 0);
  fill(seg_e, 0); // empty segments everywhere: valid, results = init

  // Baseline succeeds (all-empty segments produce init).
  segmented_reduce(in, envs, seg_b, seg_e, out, max_op{}, -5);
  {
    ::std::vector<int> h(out.size());
    out.copy_to_host(h.data());
    for (int x : h)
    {
      EXPECT(x == -5);
    }
  }

  // seg views not co-partitioned with out: refused.
  bool threw = false;
  try
  {
    segmented_reduce(in, envs, seg_b, in /* wrong region */, out, max_op{}, 0);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  // in/out shard-count mismatch: refused. (A single-piece foreign view.)
  ::std::vector<int> host_piece(3, 0);
  int* d_single = nullptr;
  cuda_safe_call(cudaMalloc(&d_single, sizeof(int) * 3));
  ::std::vector<cuda::std::span<const int>> one_piece{{d_single, 3}};
  const auto foreign_in = make_sharded_view(one_piece);
  if (P > 1)
  {
    threw = false;
    try
    {
      segmented_reduce(foreign_in, envs, seg_b, seg_e, out, max_op{}, 0);
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
  cuda_safe_call(cudaFree(d_single));

  // Synchronous form under sync_policy::forbid: refused before any work.
  threw = false;
  try
  {
    const auto fprop = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
    segmented_reduce(in, envs, seg_b, seg_e, out, max_op{}, 0, ::cuda::std::execution::env{fprop});
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  (void) out3;
}

void test_async_capture(place_group& group)
{
  const ::std::size_t P = group.size();
  auto envs             = group.envs(0);

  // Uniform segments: every shard has 8 segments of 5 elements.
  const ::std::size_t segs_per = 8, w = 5;
  ::std::vector<::std::size_t> seg_sizes(P, segs_per), in_sizes(P, segs_per * w);
  auto in    = sharded_array<float>::allocate(group, in_sizes, 0);
  auto out   = sharded_array<float>::allocate(group, seg_sizes, 0);
  auto seg_b = sharded_array<int>::allocate(group, seg_sizes, 0);
  auto seg_e = sharded_array<int>::allocate(group, seg_sizes, 0);
  fill(in, 2.0f);
  // Offsets via host upload (shard-local).
  {
    ::std::vector<int> h_b, h_e;
    for (::std::size_t g = 0; g < P; g++)
    {
      for (::std::size_t i = 0; i < segs_per; i++)
      {
        h_b.push_back(static_cast<int>(i * w));
        h_e.push_back(static_cast<int>((i + 1) * w));
      }
    }
    seg_b.copy_from_host(h_b.data());
    seg_e.copy_from_host(h_e.data());
  }

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));

  const auto cprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}};
  const auto ce    = ::cuda::std::execution::env{cprop};
  // Lane-ordered under capture: fork the lanes from the origin once, enqueue
  // in lane order, and join the lanes back with the stream barrier.
  out.fork_from(origin);
  segmented_reduce(in, envs, seg_b, seg_e, out, sum_op{}, 0.0f, ce);
  barrier(envs, ::cuda::stream_ref{origin});

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  ::std::vector<float> h_out(out.size());
  out.copy_to_host(h_out.data());
  for (float x : h_out)
  {
    EXPECT(x == 2.0f * static_cast<float>(w));
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_ragged_correctness(group);
  test_refusals(group);
  test_async_capture(group);

  printf("segmented_reduce: all tests passed\n");
  return 0;
}
