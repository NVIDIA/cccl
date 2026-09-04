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
 * @brief Correctness of the size-mutating sharded algorithms `copy_if` /
 *        `remove_if` / `filter` and `unique` against host references, over
 *        multiple places: per-shard compaction, shard-size and offset
 *        bookkeeping (including shards whose result is empty), the
 *        cross-shard boundary trim of `unique`, and the contract that
 *        size-mutating algorithms REFUSE contiguous (VMM-backed) arrays with
 *        `std::invalid_argument`.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::cuda_try;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct is_even
{
  __host__ __device__ bool operator()(long long x) const
  {
    return x % 2 == 0;
  }
};

struct less_than
{
  long long bound;

  __host__ __device__ bool operator()(long long x) const
  {
    return x < bound;
  }
};

void test_copy_if(place_group& group)
{
  const size_t n = 500009;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL); // 0 .. n-1

  const size_t kept = copy_if(data, is_even{});
  EXPECT(kept == (n + 1) / 2);
  EXPECT(data.size() == kept);
  EXPECT(data.validate());

  // Per-shard compaction is stable and shards stay ordered, so the logical
  // array equals the host copy_if of the original input
  ::std::vector<long long> host(kept);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < kept; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i));
  }

  // Capacities are unchanged: the buffers can be reused at full size
  EXPECT(data.total_capacity() == n);
  data.reset_sizes_to_capacity();
  EXPECT(data.size() == n);

  // Empty array
  sharded_array<long long> empty;
  EXPECT(copy_if(empty, is_even{}) == 0UL);
}

void test_copy_if_empty_result_shards(place_group& group)
{
  const size_t n = 300000;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL); // values == global indices

  // Keep only values inside the first half of shard 0: every other shard
  // compacts to an EMPTY result
  const long long bound = static_cast<long long>(data.shard(0).size / 2);
  const size_t kept     = copy_if(data, less_than{bound});
  EXPECT(kept == static_cast<size_t>(bound));
  EXPECT(data.size() == kept);
  EXPECT(data.validate());
  for (size_t g = 1; g < data.num_shards(); g++)
  {
    EXPECT(data.shard(g).size == 0UL);
  }

  ::std::vector<long long> host(kept);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < kept; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i));
  }

  // Keep nothing: every shard compacts to empty
  data.reset_sizes_to_capacity();
  iota(data, 0LL);
  EXPECT(copy_if(data, less_than{0}) == 0UL);
  EXPECT(data.size() == 0UL);
  EXPECT(data.validate());
}

void test_remove_if_and_filter(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL);

  // remove_if is the inverse of copy_if: drop the evens, keep the odds
  const size_t kept = remove_if(data, is_even{});
  EXPECT(kept == n / 2);
  ::std::vector<long long> host(kept);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < kept; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i) + 1);
  }

  // filter is an alias for copy_if
  auto other = sharded_array<long long>::allocate(group, n);
  iota(other, 0LL);
  EXPECT(filter(other, is_even{}) == (n + 1) / 2);
}

void test_unique_cross_shard_boundary(place_group& group)
{
  // Runs of 7 equal values; 7 does not divide the shard sizes, so runs
  // straddle shard boundaries and local per-shard unique alone would keep a
  // duplicate at each straddled boundary — the boundary trim must drop it
  const size_t n = 240007;
  auto data      = sharded_array<long long>::allocate(group, n);
  ::std::vector<long long> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<long long>(i / 7);
  }
  data.copy_from_host(host.data());

  // Host reference: std::unique over the logical array
  ::std::vector<long long> ref(host);
  ref.erase(::std::unique(ref.begin(), ref.end()), ref.end());

  const size_t u = unique(data);
  EXPECT(u == ref.size());
  EXPECT(data.size() == u);
  EXPECT(data.validate());

  ::std::vector<long long> result(u);
  data.copy_to_host(result.data());
  for (size_t i = 0; i < u; i++)
  {
    EXPECT(result[i] == ref[i]);
  }

  // Degenerate boundary case: ONE value everywhere. Every shard collapses to
  // a single element locally, then the boundary trim must chain across every
  // shard boundary, leaving exactly one element in the whole array.
  auto same = sharded_array<long long>::allocate(group, 100000);
  fill(same, 42LL);
  EXPECT(unique(same) == 1UL);
  EXPECT(same.size() == 1UL);
  long long only = 0;
  same.copy_to_host(&only);
  EXPECT(only == 42LL);

  // Empty array
  sharded_array<long long> empty;
  EXPECT(unique(empty) == 0UL);
}

// Out-of-place selection: source untouched, destination ragged, capacity
// contract enforced at entry, self-bound and explicit-envs forms agree.
void test_copy_if_out_of_place(place_group& group)
{
  const size_t n = 100003;
  auto src       = sharded_array<long long>::allocate(group, n);
  iota(src, 0LL);
  const auto src_view = src.slice(0, n); // non-owning view over the source

  // Destination with capacity == source size per shard (worst case).
  ::std::vector<size_t> caps(src.num_shards());
  for (size_t g = 0; g < src.num_shards(); g++)
  {
    caps[g] = static_cast<size_t>(src.shard(g).size);
  }
  auto dst = sharded_array<long long>::allocate(group, caps, 0);

  // Self-bound form.
  const size_t kept = copy_if(src_view, dst, is_even{});
  EXPECT(kept == (n + 1) / 2);
  EXPECT(dst.size() == kept);
  EXPECT(validate(dst));

  // Source untouched.
  EXPECT(src.size() == n);
  {
    ::std::vector<long long> h(n);
    src.copy_to_host(h.data());
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(h[i] == static_cast<long long>(i));
    }
  }
  // Destination holds exactly the even values, in order.
  {
    ::std::vector<long long> h(kept);
    dst.copy_to_host(h.data());
    // Per shard, the evens of that shard's range, concatenated in shard order.
    size_t k = 0;
    for (size_t g = 0; g < src.num_shards(); g++)
    {
      const auto& sh = src.shard(g);
      for (size_t i = 0; i < static_cast<size_t>(sh.size); i++)
      {
        const long long v = static_cast<long long>(sh.global_offset + i);
        if (v % 2 == 0)
        {
          EXPECT(h[k++] == v);
        }
      }
    }
    EXPECT(k == kept);
  }

  // Explicit-envs form agrees (rerun into the same destination: prior ragged
  // sizes are irrelevant by contract).
  auto envs          = default_envs(dst);
  const size_t kept2 = copy_if(src_view, envs, dst, is_even{});
  EXPECT(kept2 == kept);

  // Capacity refusal: a destination with a too-small shard refuses at entry
  // and stays untouched.
  ::std::vector<size_t> small(src.num_shards(), 1);
  auto tiny = sharded_array<long long>::allocate(group, small, 0);
  fill(tiny, -7LL);
  bool threw = false;
  try
  {
    (void) copy_if(src_view, tiny, is_even{});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(tiny.size() == src.num_shards()); // sizes unchanged
  {
    ::std::vector<long long> h(tiny.size());
    tiny.copy_to_host(h.data());
    for (long long v : h)
    {
      EXPECT(v == -7LL); // contents unchanged: refusal preceded any move
    }
  }

  // forbid: refused before any work.
  threw                  = false;
  const auto forbid_prop = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
  try
  {
    (void) copy_if(src_view, dst, is_even{}, ::cuda::std::execution::env{forbid_prop});
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_size_mutators_refuse_contiguous(place_group& group)
{
  // THE CONTRACT: a contiguous array is one VA range read as one array
  // through contiguous_data(); shrinking shard sizes would leave gaps between
  // shards' valid elements, and compacting across the gaps would migrate
  // elements across the requested placement — so the size-mutating
  // algorithms must refuse with std::invalid_argument, leaving the array
  // untouched.
  const size_t n = (1 << 20) + 99;
  auto data      = sharded_array<long long>::allocate_contiguous(group, n);
  iota(data, 0LL);

  bool threw = false;
  try
  {
    (void) copy_if(data, is_even{});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try
  {
    (void) unique(data);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try
  {
    (void) remove_if(data, is_even{});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  // The refused calls left the array untouched
  EXPECT(data.size() == n);
  EXPECT(data.validate());
  EXPECT(count(data, 1LL) == 1UL); // data intact, read-only path still fine
}
} // namespace

int main()
{
  cuda_try(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_copy_if(group);
  test_copy_if_empty_result_shards(group);
  test_copy_if_out_of_place(group);
  test_remove_if_and_filter(group);
  test_unique_cross_shard_boundary(group);

  if (contiguous_backing_supported())
  {
    test_size_mutators_refuse_contiguous(group);
  }
  else
  {
    printf("VMM not supported on this device, skipping contiguous-array tests.\n");
  }

  return 0;
}
