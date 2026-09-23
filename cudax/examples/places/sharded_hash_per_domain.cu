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
 * @brief One hash table per locality domain: build it where its data lives,
 *        probe it from the same place, drive both with `for_each_shard`.
 *
 * The replicated-table shape of a NUMA-aware hash join: the build side is a
 * sharded array (one shard per locality domain), and every domain builds ITS
 * OWN `cuda::experimental::cuco::fixed_capacity_map` from its shard, with the
 * table's storage placed on that domain through the group's per-place memory
 * resource. Probes issued by a domain then only ever touch a table in the
 * domain's own memory. Nothing crosses the locality boundary.
 *
 * What this shows about the sharded tier:
 *  1. placement of a container the tier does not know about needs only an
 *     allocator hook: `fixed_capacity_map`'s `_MemoryResource` parameter takes
 *     `place_memory_resource` as-is (it is `device_accessible`);
 *  2. per-shard work the tier has no verb for — construct a map, insert into
 *     it, probe it, reduce a per-shard count — is spelled with
 *     `for_each_shard`: the body only enqueues on the shard's stream, in the
 *     shard's execution context; the driver owns the ordering, skips empty
 *     shards, and drains the lanes (synchronous form) or leaves them
 *     lane-ordered (asynchronous form);
 *  3. three body arities: `(g, shard, env)` when the body hands the shard's
 *     environment to CUB (stream + memory resource for scratch),
 *     `(g, shard, stream)` when it needs the shard index, `(shard, stream)`
 *     otherwise. The body only enqueues: results go to device slots, the
 *     host reads them after `barrier(envs)`.
 *
 * Data: key i of the build side is the odd number 2i+1, stored with payload i.
 * Probe i asks for 2i+1 when i is odd (a hit, in the probing domain's own
 * table) and for the even number 2i otherwise (a miss). The expected hit count
 * per domain is therefore the number of odd global indices in its shard.
 */

#include <cub/device/device_reduce.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/sharded.cuh>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;
using cuda::experimental::places::place_memory_resource;
namespace xcuco = cuda::experimental::cuco;

// Build-side row i -> (key 2i+1, payload i), produced on the fly from the
// global index: no key array has to be materialized.
struct key_value_of
{
  __host__ __device__ ::cuda::std::pair<int, int> operator()(::std::size_t i) const
  {
    return {static_cast<int>(2 * i + 1), static_cast<int>(i)};
  }
};

// Probe i: hit on odd i (its own key), miss on even i (an even number).
struct probe_of
{
  __host__ __device__ int operator()(::std::size_t i) const
  {
    return static_cast<int>((i & 1) ? 2 * i + 1 : 2 * i);
  }
};

int main()
{
  auto group            = place_group{make_locality_domain_grid()};
  const ::std::size_t P = group.size();
  printf("place_group with %zu place(s)\n", P);

  // The probe side and the hit flags: one shard per domain, each on its
  // domain's memory. (The build side needs no array at all, see key_value_of.)
  const ::std::size_t n = ::std::size_t{1} << 24;
  auto probes           = sharded_array<int>::allocate(group, n);
  auto found            = sharded_array<int>::allocate(group, n); // 0/1 per probe
  tabulate(probes, probe_of{});
  const auto envs = default_envs(probes);

  // One table per domain. fixed_capacity_map's _MemoryResource is
  // place_memory_resource: the slots live where the domain's data lives.
  using map_t = xcuco::fixed_capacity_map<
    int,
    int,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<int>,
    xcuco::linear_probing<4, ::cuda::hash<int>>,
    1,
    place_memory_resource>;
  ::std::vector<::std::unique_ptr<map_t>> tables(P);

  // 1. BUILD: for every shard, on its stream, construct the table on the
  //    shard's place and insert the shard's rows. (g, shard, stream) arity:
  //    the body needs the shard index to pick its table.
  for_each_shard(probes, envs, [&](::std::size_t g, const auto& d, cudaStream_t s) {
    const ::cuda::stream_ref stream{s};
    tables[g] = ::std::make_unique<map_t>(
      stream, group.memory_resource(g), /*capacity*/ 2 * d.size, xcuco::empty_key<int>{-1}, xcuco::empty_value<int>{-1});
    auto rows = thrust::make_transform_iterator(thrust::make_counting_iterator(d.global_offset), key_value_of{});
    tables[g]->insert_async(stream, rows, rows + d.size);
  }); // synchronous form: every table is built when this returns

  for (::std::size_t g = 0; g < P; g++)
  {
    printf("  domain %zu: exec=%s  table storage=%s  capacity=%zu  rows=%zu\n",
           g,
           group.place(g).to_string().c_str(),
           group.memory_resource(g).place().to_string().c_str(),
           static_cast<::std::size_t>(tables[g]->capacity()),
           probes.shard(g).size);
  }

  // 2. PROBE: each domain queries its own table with its own probe shard.
  //    (g, shard, stream) again; the output is the co-partitioned `found`.
  for_each_shard(probes, envs, [&](::std::size_t g, const auto& d, cudaStream_t s) {
    tables[g]->contains_async(::cuda::stream_ref{s}, d.data, d.data + d.size, found.shard(g).data);
  });

  // 3. PER-DOMAIN COUNT: a reduction per shard whose P results the caller keeps
  //    (`reduce` without its fold). (g, shard, env) arity: the shard
  //    environment IS a CUB environment — `cub::DeviceReduce::Sum(in, out, n,
  //    env)` runs on its stream and draws its temp storage from its memory
  //    resource, so scratch lands on the shard's place. The result goes to a
  //    device slot, never to the host inside the body: a value-returning
  //    reduce would synchronize the stream and serialize the lanes. The host
  //    copies are enqueued on the lanes and read after barrier().
  ::std::vector<long long*> slots(P, nullptr);
  ::std::vector<long long> hits(P, -1);
  for_each_shard(found, envs, [&](::std::size_t g, const auto& d, const auto& env) {
    const ::cuda::stream_ref s = ::cuda::get_stream(env);
    auto mr                    = ::cuda::mr::get_memory_resource(env);
    slots[g]                   = static_cast<long long*>(mr.allocate(s, sizeof(long long), alignof(long long)));
    cuda_safe_call(cub::DeviceReduce::Sum(d.data, slots[g], d.size, env));
    cuda_safe_call(cudaMemcpyAsync(&hits[g], slots[g], sizeof(long long), cudaMemcpyDeviceToHost, s.get()));
  });
  barrier(envs);

  bool ok = true;
  for (::std::size_t g = 0; g < P; g++)
  {
    const auto& d = found.shard(g);
    // odd global indices in [global_offset, global_offset + size)
    const long long first_odd = static_cast<long long>(d.global_offset | 1);
    const long long last      = static_cast<long long>(d.global_offset + d.size);
    const long long expected  = first_odd < last ? (last - first_odd + 1) / 2 : 0;
    printf("  domain %zu: %lld hits (expected %lld)\n", g, hits[g], expected);
    ok      = ok && hits[g] == expected;
    auto mr = ::cuda::mr::get_memory_resource(envs[g]);
    mr.deallocate(::cuda::get_stream(envs[g]), slots[g], sizeof(long long), alignof(long long));
  }
  barrier(envs);
  tables.clear(); // each table frees its slots on the stream it was given

  if (!ok)
  {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
