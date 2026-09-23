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
 * @brief A morsel-driven hash join across locality domains: filter morsels
 *        where they live, combine through ONE shared hash table in device
 *        memory, probe from morsels again.
 *
 * The build phase of a NUMA-aware hash join as in Leis et al., "Morsel-Driven
 * Parallelism" (SIGMOD 2014), Figure 3, spelled with the sharded tier:
 *
 *  - a **morsel** is a shard: a contiguous piece of a table tagged with the
 *    locality domain that owns its memory. Here each domain holds K morsels
 *    of the build table T and K morsels of the probe table R, all adopted
 *    into one `sharded_array` per table (many shards per place);
 *  - **phase 1** filters T morsel-wise, in place, into local storage:
 *    `select_if` runs on every morsel on its domain's stream and commits the
 *    surviving sizes, so the exact survivor count is known before the table
 *    is sized — the paper's reason for splitting the build in two;
 *  - **phase 2** builds one **shared** `fixed_capacity_map` at exact size.
 *    Its storage is plain device memory (`data_place::device(0)`), NOT a
 *    locality domain: it is the structure every domain writes and reads, so it
 *    is deliberately left unlocalized, the way the paper interleaves its hash
 *    table across sockets. Every morsel of every domain inserts into it from
 *    its own stream (`for_each_shard`);
 *  - the **probe pipeline** streams R morsel-wise from each domain through
 *    the shared table and writes its flags into local storage; a per-morsel
 *    count (CUB on the shard environment) gives the per-domain totals.
 *
 * Streaming data (T, R, the flags) stays in the domain that owns it. The only
 * cross-domain traffic is the shared table.
 *
 * Data: row i of T has key 2i+1; the filter keeps keys not divisible by 3.
 * Probe i asks for key 2i+1, so it hits iff (2i+1) % 3 != 0.
 */

#include <cub/device/device_reduce.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/sharded.cuh>

#include <cstddef>
#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::data_place;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;
using cuda::experimental::places::place_memory_resource;
namespace xcuco = cuda::experimental::cuco;

struct key_of // row i -> key 2i+1
{
  __host__ __device__ int operator()(::std::size_t i) const
  {
    return static_cast<int>(2 * i + 1);
  }
};
struct keep // the filter: drop every third key
{
  __host__ __device__ bool operator()(int key) const
  {
    return key % 3 != 0;
  }
};
struct to_pair // build row -> (key, payload = key)
{
  __host__ __device__ ::cuda::std::pair<int, int> operator()(int key) const
  {
    return {key, key};
  }
};

// Cut one buffer per domain into K morsels, all tagged with that domain, and
// adopt them as a single sharded view (P*K shards, K per place).
struct morsel_table
{
  ::std::vector<int*> buffers; // one allocation per domain
  ::std::size_t rows_per_domain = 0;
  sharded_array<int> view;

  morsel_table(place_group& group, ::std::size_t n, ::std::size_t K)
      : buffers(group.size(), nullptr)
      , rows_per_domain(n / group.size())
      , view(sharded_array<int>::adopt(cut(group, K)))
  {}
  ::std::vector<shard<int>> cut(place_group& group, ::std::size_t K)
  {
    const ::std::size_t P = group.size(), morsel = rows_per_domain / K;
    ::std::vector<shard<int>> shards;
    for (::std::size_t g = 0; g < P; g++)
    {
      const data_place dp     = group.place(g).affine_data_place();
      const cudaStream_t lane = group.get_stream(g, 0);
      buffers[g] = static_cast<int*>(dp.allocate(static_cast<::std::ptrdiff_t>(rows_per_domain * sizeof(int)), lane));
      for (::std::size_t m = 0; m < K; m++)
      {
        shard<int> s;
        s.data          = buffers[g] + m * morsel;
        s.size          = morsel;
        s.capacity      = morsel;
        s.global_offset = (g * K + m) * morsel;
        s.place         = dp;
        s.exec          = group.place(g);
        s.stream        = lane; // the K morsels of a domain share its lane
        shards.push_back(s);
      }
    }
    return shards;
  }
  void release(place_group& group)
  {
    for (::std::size_t g = 0; g < group.size(); g++)
    {
      group.place(g).affine_data_place().deallocate(buffers[g], rows_per_domain * sizeof(int), group.get_stream(g, 0));
    }
  }
};

int main()
{
  auto group            = place_group{make_locality_domain_grid()};
  const ::std::size_t P = group.size();
  const ::std::size_t K = 4; // morsels per domain
  const ::std::size_t n = ::std::size_t{1} << 22;
  printf("place_group with %zu place(s), %zu morsels per place, %zu rows\n", P, K, n);

  // The build table T and the probe table R, morsel-wise on the domains.
  morsel_table T{group, n, K}, R{group, n, K}, flags{group, n, K};
  tabulate(T.view, key_of{});
  tabulate(R.view, key_of{});
  const auto t_envs = default_envs(T.view);
  const auto r_envs = default_envs(R.view);

  // ---- Phase 1: filter T morsel-wise, in place, into local storage --------
  // Every morsel is compacted on its domain's stream; the surviving sizes are
  // committed atomically. Nothing moves between domains.
  const ::std::size_t survivors = select_if(T.view, t_envs, keep{});
  printf("phase 1: %zu of %zu build rows survive the filter\n", survivors, n);
  for (::std::size_t s = 0; s < T.view.num_shards(); s++)
  {
    printf("  morsel %2zu (domain %zu): %zu rows\n", s, s / K, T.view.shard(s).size);
  }

  // ---- Phase 2: ONE shared hash table, exact size, in device memory --------
  // The table is what every domain touches, so it is not localized: its
  // storage is data_place::device(0). (Swap in a locality domain or a
  // composite place to experiment; nothing else changes.)
  using map_t = xcuco::fixed_capacity_map<
    int,
    int,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<int>,
    xcuco::linear_probing<4, ::cuda::hash<int>>,
    1,
    place_memory_resource>;
  place_memory_resource shared_mr{data_place::device(0)};
  const cudaStream_t s0 = group.get_stream(0, 0);
  map_t table{::cuda::stream_ref{s0},
              shared_mr,
              /*capacity*/ 2 * survivors,
              xcuco::empty_key<int>{-1},
              xcuco::empty_value<int>{-1}};
  cuda_safe_call(cudaStreamSynchronize(s0)); // the table's storage is initialized before any lane inserts
  printf("phase 2: shared table storage=%s, capacity %zu\n",
         shared_mr.place().to_string().c_str(),
         static_cast<::std::size_t>(table.capacity()));

  // Every morsel of every domain inserts its survivors into the one table,
  // from its own stream. Device-scope atomics make the concurrent inserts
  // safe; the synchronous form drains all lanes before we go on.
  for_each_shard(T.view, t_envs, [&](const auto& d, cudaStream_t s) {
    auto rows = thrust::make_transform_iterator(d.data, to_pair{});
    table.insert_async(::cuda::stream_ref{s}, rows, rows + d.size);
  });

  // ---- Probe pipeline: R morsel-wise through the shared table --------------
  // Each morsel probes from its domain and writes its flags into the
  // co-partitioned local morsel of `flags`.
  for_each_shard(R.view, r_envs, [&](::std::size_t g, const auto& d, cudaStream_t s) {
    table.contains_async(::cuda::stream_ref{s}, d.data, d.data + d.size, flags.view.shard(g).data);
  });

  // Per-morsel hit counts: CUB on the shard environment, result to a device
  // slot on the shard's place, host copies read after barrier().
  const ::std::size_t M = flags.view.num_shards();
  ::std::vector<long long*> slots(M, nullptr);
  ::std::vector<long long> hits(M, -1);
  for_each_shard(flags.view, r_envs, [&](::std::size_t g, const auto& d, const auto& env) {
    const ::cuda::stream_ref s = ::cuda::get_stream(env);
    auto mr                    = ::cuda::mr::get_memory_resource(env);
    slots[g]                   = static_cast<long long*>(mr.allocate(s, sizeof(long long), alignof(long long)));
    cuda_safe_call(cub::DeviceReduce::Sum(d.data, slots[g], d.size, env));
    cuda_safe_call(cudaMemcpyAsync(&hits[g], slots[g], sizeof(long long), cudaMemcpyDeviceToHost, s.get()));
  });
  barrier(r_envs);

  // Expected: probe i hits iff key 2i+1 is not a multiple of 3.
  bool ok = true;
  for (::std::size_t g = 0; g < P; g++)
  {
    long long got = 0, want = 0;
    for (::std::size_t m = 0; m < K; m++)
    {
      const auto& d = flags.view.shard(g * K + m);
      got += hits[g * K + m];
      for (::std::size_t i = d.global_offset; i < d.global_offset + d.size; i++)
      {
        want += (2 * i + 1) % 3 != 0;
      }
      auto mr = ::cuda::mr::get_memory_resource(r_envs[g * K + m]);
      mr.deallocate(::cuda::get_stream(r_envs[g * K + m]), slots[g * K + m], sizeof(long long), alignof(long long));
    }
    printf("probe: domain %zu: %lld hits (expected %lld)\n", g, got, want);
    ok = ok && got == want;
  }
  barrier(r_envs);

  T.release(group);
  R.release(group);
  flags.release(group);
  if (!ok)
  {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
