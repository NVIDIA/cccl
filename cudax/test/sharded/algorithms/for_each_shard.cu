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
 * @brief `for_each_shard`: the map family's driver as a public verb. Covers
 *        the three body arities, a raw per-shard kernel, a per-shard
 *        reduction returning P host values (the first half of `reduce`,
 *        written by the caller: CUB on the shard environment, result to a
 *        device slot, no host sync in the body), the skip of empty shards, and the asynchronous
 *        lane-ordered form followed by `barrier`.
 */

#include <cub/device/device_reduce.cuh>

#include <cuda/stream>

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
__global__ void stamp_kernel(long long* p, size_t n, long long tag)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    p[i] = tag + static_cast<long long>(i);
  }
}

// (size_t, shard, cudaStream_t): a hand-written kernel per shard, on the
// shard's stream, tagged by the shard index; verified against the view's
// own shard boundaries.
void test_raw_kernel_per_shard(place_group& group)
{
  const size_t n  = 100003;
  auto data       = sharded_array<long long>::allocate(group, n);
  const auto envs = default_envs(data);

  for_each_shard(data, envs, [](size_t g, const auto& d, cudaStream_t s) {
    const unsigned blocks = static_cast<unsigned>((d.size + 255) / 256);
    stamp_kernel<<<blocks, 256, 0, s>>>(d.data, d.size, 1000000LL * static_cast<long long>(g));
    cuda_safe_call(cudaGetLastError());
  }); // synchronous form: returns with every shard stream drained

  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    const auto& d = data.shard(g);
    for (size_t i = 0; i < d.size; i++)
    {
      EXPECT(host[d.global_offset + i] == 1000000LL * static_cast<long long>(g) + static_cast<long long>(i));
    }
  }
}

// (size_t, shard, env): per-shard reduction into a device slot drawn from
// the shard environment's memory resource, P host results after one
// barrier — `reduce` without its fold, spelled by the caller.
void test_per_shard_reduce_with_env(place_group& group)
{
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 1LL); // 1, 2, 3, ...
  const auto envs = default_envs(data);
  const size_t P  = data.num_shards();

  ::std::vector<long long*> slots(P, nullptr);
  ::std::vector<long long> totals(P, -1);

  for_each_shard(data, envs, [&](size_t g, const auto& d, const auto& env) {
    const ::cuda::stream_ref s = ::cuda::get_stream(env);
    auto mr                    = ::cuda::mr::get_memory_resource(env);
    slots[g]                   = static_cast<long long*>(mr.allocate(s, sizeof(long long), alignof(long long)));
    // The shard env is a CUB env: stream + memory resource for temp storage.
    // Result to a device slot, no host sync inside the body.
    cuda_safe_call(cub::DeviceReduce::Sum(d.data, slots[g], d.size, env));
    cuda_safe_call(cudaMemcpyAsync(&totals[g], slots[g], sizeof(long long), cudaMemcpyDeviceToHost, s.get()));
  });
  // The synchronous form has already drained the lanes; a barrier is what
  // the asynchronous form would need here. Both are legal.
  barrier(envs);

  long long grand = 0;
  for (size_t g = 0; g < P; g++)
  {
    const auto& d        = data.shard(g);
    const long long lo   = static_cast<long long>(d.global_offset) + 1;
    const long long hi   = static_cast<long long>(d.global_offset + d.size);
    const long long want = (hi * (hi + 1)) / 2 - ((lo - 1) * lo) / 2;
    EXPECT(totals[g] == want);
    grand += totals[g];
    auto mr = ::cuda::mr::get_memory_resource(envs[g]);
    mr.deallocate(::cuda::get_stream(envs[g]), slots[g], sizeof(long long), alignof(long long));
  }
  EXPECT(grand == static_cast<long long>(n) * static_cast<long long>(n + 1) / 2);
  barrier(envs);
}

// (shard, cudaStream_t) + self-bound form + asynchronous lane-ordered call:
// the body enqueues only; the caller seals with barrier(envs).
void test_two_arg_body_async(place_group& group)
{
  const size_t n = 4099;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(data, 0LL);
  const auto envs = default_envs(data);

  ::cuda::stream caller{::cuda::device_ref{0}};
  const auto caller_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{caller}};
  const auto caller_env  = ::cuda::std::execution::env{caller_prop};

  for_each_shard(
    data,
    [](const auto& d, cudaStream_t s) {
      stamp_kernel<<<static_cast<unsigned>((d.size + 255) / 256), 256, 0, s>>>(d.data, d.size, 7LL);
      cuda_safe_call(cudaGetLastError());
    },
    caller_env); // lane-ordered: nothing waits on the caller stream
  barrier(envs);

  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    const auto& d = data.shard(g);
    for (size_t i = 0; i < d.size; i++)
    {
      EXPECT(host[d.global_offset + i] == 7LL + static_cast<long long>(i));
    }
  }
}

// Empty shards are skipped: a foreign view with a hole in the middle.
struct foreign_env
{
  cudaStream_t s;
  ::cuda::stream_ref get_stream() const noexcept
  {
    return ::cuda::stream_ref{s};
  }
};
struct holey_view
{
  ::std::vector<basic_shard_view<int, int>> shards_;
  ::std::vector<foreign_env> envs_;
  size_t num_shards() const
  {
    return shards_.size();
  }
  const basic_shard_view<int, int>& shard(size_t i) const
  {
    return shards_[i];
  }
};
::std::vector<foreign_env> default_envs(const holey_view& v)
{
  return v.envs_;
}
static_assert(sharded_view<holey_view>);
static_assert(self_bound<holey_view>);

void test_empty_shards_skipped()
{
  int* buf = nullptr;
  cuda_safe_call(cudaMalloc(&buf, 16 * sizeof(int)));
  cudaStream_t s;
  cuda_safe_call(cudaStreamCreate(&s));
  holey_view v;
  v.shards_ = {{buf, 8, 0, 0}, {buf + 8, 0, 8, 0}, {buf + 8, 8, 8, 0}};
  v.envs_   = {{s}, {s}, {s}};

  ::std::vector<size_t> visited;
  for_each_shard(v, [&](size_t g, const auto& d, cudaStream_t) {
    EXPECT(d.size != 0);
    visited.push_back(g);
  });
  EXPECT(visited.size() == 2);
  EXPECT(visited[0] == 0);
  EXPECT(visited[1] == 2);

  cuda_safe_call(cudaStreamDestroy(s));
  cuda_safe_call(cudaFree(buf));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_raw_kernel_per_shard(group);
  test_per_shard_reduce_with_env(group);
  test_two_arg_body_async(group);
  test_empty_shards_skipped();

  return 0;
}
