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
 * @brief Sharded reduction over the locality domains of the machine.
 *
 * Mapping-tier example (no task framework required):
 *  1. build a `place_group` — one place per locality domain of every device
 *     (whole devices where domains are unsupported),
 *  2. allocate a `sharded_array` over the group — each shard lives on its
 *     domain's memory, with a reference stream from the group's pools,
 *  3. run sharded algorithms: each place executes the device-scope primitive
 *     (CUB) on its shard, and the per-place results are combined — the same
 *     local-compute-plus-combine structure CUDA already uses from warps to
 *     blocks to devices, extended one scope further.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstddef>
#include <cstdio>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

int main()
{
  // One execution place per locality domain, with lazily created per-place
  // stream pools and memory resources
  auto group = place_group{make_locality_domain_grid()};
  printf("place_group with %zu place(s)\n", group.size());

  // 256M values, distributed evenly: shard i lives on place i
  const std::size_t n = std::size_t{1} << 28;
  auto data           = sharded_array<long long>::allocate(group, n);

  // data[i] = i + 1 (global index), computed by each place on its shard
  iota(data, 1LL);

  // Per-place CUB reduction + combine across places
  const long long total    = sum(data);
  const long long expected = static_cast<long long>(n) * (static_cast<long long>(n) + 1) / 2;

  printf("sum(1..%zu) = %lld (expected %lld)\n", n, total, expected);
  printf("min = %lld, max = %lld\n", min(data), max(data));

  if (total != expected)
  {
    printf("FAILED\n");
    return 1;
  }

  printf("PASSED\n");
  return 0;
}
