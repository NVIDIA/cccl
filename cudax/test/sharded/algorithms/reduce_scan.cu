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
 * @brief Correctness of the sharded reduce / scan / adjacent_difference
 *        algorithms against host references, over multiple places: per-place
 *        CUB primitive plus cross-place combine.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <limits>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
struct max_op
{
  __host__ __device__ long long operator()(long long a, long long b) const
  {
    return a > b ? a : b;
  }
};

void test_reduce(place_group& group)
{
  const size_t n = 1000001;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(group, data, 1LL); // 1..n

  const long long expected_sum = static_cast<long long>(n) * (static_cast<long long>(n) + 1) / 2;
  EXPECT(sum(group, data) == expected_sum);
  EXPECT(min(group, data) == 1LL);
  EXPECT(max(group, data) == static_cast<long long>(n));

  // Custom operator through the generic entry point
  EXPECT(reduce(group, data, max_op{}, 0LL) == static_cast<long long>(n));

  // Empty array returns the initial value
  sharded_array<long long> empty;
  EXPECT(reduce(group, empty, max_op{}, -7LL) == -7LL);
}

void test_inclusive_scan(place_group& group)
{
  const size_t n = 262147;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(group, data, 1LL);

  inclusive_scan(group, data); // 1, 2, 3, ..., n
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }

  // Custom operator: running maximum of iota is iota itself
  iota(group, data, 0LL);
  inclusive_scan(group, data, max_op{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i));
  }
}

void test_exclusive_scan(place_group& group)
{
  const size_t n = 131075;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(group, data, 2LL);

  exclusive_scan(group, data); // 0, 2, 4, ..., 2*(n-1)
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i));
  }

  // inclusive_sum / exclusive_sum aliases
  fill(group, data, 1LL);
  inclusive_sum(group, data);
  data.copy_to_host(host.data());
  EXPECT(host[n - 1] == static_cast<long long>(n));

  fill(group, data, 1LL);
  exclusive_sum(group, data);
  data.copy_to_host(host.data());
  EXPECT(host[0] == 0LL);
  EXPECT(host[n - 1] == static_cast<long long>(n) - 1);
}

void test_adjacent_difference(place_group& group)
{
  const size_t n = 100000;
  auto input     = sharded_array<long long>::allocate(group, n);
  auto output    = sharded_array<long long>::allocate_like(input);

  // input[i] = i*i; diff[i] = i*i - (i-1)^2 = 2i - 1 (and diff[0] = 0)
  ::std::vector<long long> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<long long>(i) * static_cast<long long>(i);
  }
  input.copy_from_host(host.data());

  adjacent_difference(group, input, output);
  output.copy_to_host(host.data());

  EXPECT(host[0] == 0LL); // first element kept as-is
  for (size_t i = 1; i < n; i++)
  {
    if (host[i] != 2 * static_cast<long long>(i) - 1)
    {
    }
    EXPECT(host[i] == 2 * static_cast<long long>(i) - 1);
  }

  // The cross-shard boundary elements are exercised whenever the group has
  // more than one place (indices at shard boundaries take the prev_last path)
}

void test_scan_semantics(place_group& group)
{
  // Non-additive operator across shards: inclusive product needs NO identity
  // and must fold the true cross-shard prefix (regression: a zero seed used
  // to collapse every prefix for multiplies).
  {
    const size_t n = 64; // 2^64 would overflow; use values of 1 with a few 2s
    auto data      = sharded_array<long long>::allocate(group, n);
    fill(group, data, 1LL);
    ::std::vector<long long> host(n, 1);
    host[3]     = 2;
    host[n / 2] = 2;
    host[n - 1] = 2; // three 2s, spread across shards
    data.copy_from_host(host.data());
    inclusive_scan(group, data, ::cuda::std::multiplies<long long>{});
    data.copy_to_host(host.data());
    long long running = 1;
    ::std::vector<long long> ref(n, 1);
    ref[3]     = 2;
    ref[n / 2] = 2;
    ref[n - 1] = 2;
    for (size_t i = 0; i < n; i++)
    {
      running *= ref[i];
      EXPECT(host[i] == running);
    }
  }

  // Exclusive scan with a non-zero init folds the init exactly ONCE into the
  // global sequence (regression: it used to be folded per shard).
  {
    const size_t n = 4099;
    auto data      = sharded_array<long long>::allocate(group, n);
    fill(group, data, 1LL);
    exclusive_scan(group, data, 5LL); // 5, 6, 7, ...
    ::std::vector<long long> host(n);
    data.copy_to_host(host.data());
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(host[i] == 5LL + static_cast<long long>(i));
    }
  }

  // Custom-op exclusive scan takes init AND identity explicitly.
  {
    const size_t n = 1025;
    auto data      = sharded_array<long long>::allocate(group, n);
    iota(group, data, 0LL);
    // running max, seeded with 7: out[i] = max(7, 0, ..., i-1) = max(7, i-1)
    exclusive_scan(group, data, max_op{}, 7LL, ::std::numeric_limits<long long>::lowest());
    ::std::vector<long long> host(n);
    data.copy_to_host(host.data());
    for (size_t i = 0; i < n; i++)
    {
      const long long expected = (i == 0) ? 7LL : ::std::max(7LL, static_cast<long long>(i) - 1);
      EXPECT(host[i] == expected);
    }
  }

  // Scans cross allocation-empty shards unharmed.
  if (group.size() >= 2)
  {
    ::std::vector<size_t> sizes(group.size(), 0);
    const size_t n          = 513;
    sizes[group.size() - 1] = n; // only the LAST place holds data
    auto data               = sharded_array<long long>::allocate(group, sizes);
    fill(group, data, 1LL);
    inclusive_scan(group, data);
    ::std::vector<long long> host(n);
    data.copy_to_host(host.data());
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(host[i] == static_cast<long long>(i) + 1);
    }
  }
}

void test_adjacent_difference_empty_shard(place_group& group)
{
  if (group.size() < 2)
  {
    return;
  }
  // {nonzero, 0, ...}: the predecessor boundary must cross the empty shard.
  ::std::vector<size_t> sizes(group.size(), 0);
  const size_t n_first    = 100;
  const size_t n_last     = 101;
  sizes[0]                = n_first;
  sizes[group.size() - 1] = n_last;
  const size_t n          = n_first + n_last;

  auto input  = sharded_array<long long>::allocate(group, sizes);
  auto output = sharded_array<long long>::allocate(group, sizes);
  iota(group, input, 0LL);
  adjacent_difference(group, input, output);
  ::std::vector<long long> host(n);
  output.copy_to_host(host.data());
  EXPECT(host[0] == 0LL); // first element copied
  for (size_t i = 1; i < n; i++)
  {
    EXPECT(host[i] == 1LL); // iota differences, INCLUDING across the empty shard
  }

  // Aliasing refusal
  bool threw = false;
  try
  {
    adjacent_difference(group, input, input);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_per_shard_sync(place_group& group)
{
  const size_t n = 4096;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(group, data, 9LL, /*blocking=*/false);
  for (size_t i = 0; i < data.num_shards(); i++)
  {
    data.sync(i); // per-shard member: exec scope + synchronize
  }
  EXPECT(count(group, data, 9LL) == n);
}

void test_reduce_with_empty_shard(place_group& group)
{
  if (group.size() < 2)
  {
    return;
  }
  ::std::vector<size_t> sizes(group.size(), 0);
  const size_t n = 4097;
  sizes[0]       = n;
  auto arr       = sharded_array<long long>::allocate(group, sizes);
  fill(group, arr, 3LL);
  EXPECT(sum(group, arr) == 3LL * static_cast<long long>(n));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_reduce(group);
  test_inclusive_scan(group);
  test_exclusive_scan(group);
  test_adjacent_difference(group);
  test_reduce_with_empty_shard(group);
  test_scan_semantics(group);
  test_adjacent_difference_empty_shard(group);
  test_per_shard_sync(group);

  return 0;
}
