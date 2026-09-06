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
 * @brief The whole point, in four calls: allocate one array across every
 *        locality domain of the machine, then chain algorithms over it —
 *        a global sort, then the RAGGED ones whose per-shard result sizes
 *        are data-dependent — with each step a one-liner and placement,
 *        streams and temporaries handled by the shards' own bindings.
 *
 *        sort -> unique -> filter -> sum
 *
 *        `sort` orders the whole array (each shard keeps its boundaries; the
 *        engine loads across boundaries through the one address space the
 *        domains share). `unique` then removes adjacent duplicates —
 *        including runs that straddle a shard boundary — and `filter`
 *        keeps the odd values; both shrink each shard by a data-dependent
 *        amount and commit the new sizes atomically (offsets re-tile). Every
 *        later algorithm just sees the smaller, still-valid sharded array.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct is_odd
{
  __device__ bool operator()(long long x) const
  {
    return x % 2 != 0;
  }
};
} // namespace

int main()
{
  auto group = place_group{make_locality_domain_grid()};

  const size_t n = size_t{1} << 24; // 16M values across all domains
  auto data      = sharded_array<long long>::allocate(group, n);

  // Pseudo-random keys in [0, 200000) — many duplicates, arbitrary order.
  std::vector<long long> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<long long>((i * 2654435761ull + 12345ull) % 200000ull);
  }
  data.copy_from_host(host.data());

  sort(group, data); //  globally ascending
  const size_t distinct = unique(data); //  duplicates removed (ragged shrink)
  const size_t odds     = filter(data, is_odd{}); //  keep odd values (ragged shrink)
  const long long total = sum(data); //  sum of the surviving values

  // Host reference over the same pipeline.
  std::sort(host.begin(), host.end());
  host.erase(std::unique(host.begin(), host.end()), host.end());
  host.erase(std::remove_if(host.begin(),
                            host.end(),
                            [](long long x) {
                              return x % 2 == 0;
                            }),
             host.end());
  long long ref_sum = 0;
  for (long long v : host)
  {
    ref_sum += v;
  }

  std::printf("%zu values -> %zu distinct -> %zu odd, sum = %lld (expected %lld)\n", n, distinct, odds, total, ref_sum);

  if (distinct == 0 || odds != host.size() || total != ref_sum)
  {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED (%zu place(s))\n", group.size());
  return 0;
}
