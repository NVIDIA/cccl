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
 * @brief PROTOTYPE — K shards on P places (repeated places). A P-shard
 *        owning array is re-viewed as 2P half-shards, two per place sharing
 *        that place's stream, through the raw `adopt(vector<shard>)`. The
 *        map family (transform), the combine family (sum) and a
 *        size-mutating verb (select_if) run over the 2P view and are checked
 *        against the host. The combine family is timed on the P view (lane's
 *        cached communicators) and the 2P view (per-call communicator
 *        fallback) to expose the cost the design note predicts.
 */

#include <cuda/stream>

#include <cuda/experimental/sharded.cuh>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::place_group;

namespace
{

struct negate_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return -x;
  }
};

struct is_even
{
  __host__ __device__ bool operator()(long long x) const
  {
    return (x & 1LL) == 0;
  }
};

// Two half-shards per shard of `a`, same place / exec / stream.
sharded_array<long long> split_in_halves(sharded_array<long long>& a)
{
  ::std::vector<shard<long long>> halves;
  halves.reserve(2 * a.num_shards());
  for (size_t g = 0; g < a.num_shards(); ++g)
  {
    const auto& s   = a.shard(g);
    const size_t h  = s.size / 2;
    shard<long long> lo = s, hi = s;
    lo.size = lo.capacity = h;
    hi.data               = s.data + h;
    hi.size = hi.capacity = s.size - h;
    hi.global_offset      = s.global_offset + h;
    halves.push_back(lo);
    halves.push_back(hi);
  }
  return sharded_array<long long>::adopt(::std::move(halves));
}

void test_map_and_combine(place_group& group)
{
  const size_t n = (1u << 22) + 13; // ragged tail
  auto a         = sharded_array<long long>::allocate(group, n);
  sequence(a, default_envs(a), 0LL, 1LL); // a[i] = i
  barrier(default_envs(a));

  auto v = split_in_halves(a);
  EXPECT(v.num_shards() == 2 * group.size());

  // --- map family over the 2P view: a[i] = -i
  transform(v, negate_op{}); // default_envs(v): per-shard envs with repeated streams
  {
    ::std::vector<long long> host(n);
    a.copy_to_host(host.data());
    for (size_t i = 0; i < n; ++i)
    {
      EXPECT(host[i] == -static_cast<long long>(i));
    }
  }
  printf("  transform over %zu shards on %zu places: PASSED\n", v.num_shards(), group.size());

  // --- combine family over the 2P view
  const long long expected = -static_cast<long long>(n) * static_cast<long long>(n - 1) / 2;
  const long long s2p      = sum(v);
  EXPECT(s2p == expected);
  const long long sp = sum(a);
  EXPECT(sp == expected);
  printf("  sum over 2P shards == sum over P shards == %lld: PASSED\n", expected);

  // --- timing: P (lane communicators) vs 2P (per-call communicator)
  auto time_sum = [&](auto& arr, int reps) {
    (void) sum(arr); // warm up
    const auto t0 = ::std::chrono::steady_clock::now();
    long long acc = 0;
    for (int r = 0; r < reps; ++r)
    {
      acc += sum(arr);
    }
    const auto t1 = ::std::chrono::steady_clock::now();
    EXPECT(acc == expected * reps);
    return ::std::chrono::duration<double, ::std::micro>(t1 - t0).count() / reps;
  };
  const int reps    = 20;
  const double us_p = time_sum(a, reps);
  const double us_2 = time_sum(v, reps);
  printf("  sum wall time: P shards %.1f us/call, 2P shards %.1f us/call (ratio %.2fx)\n", us_p, us_2, us_2 / us_p);
}

void test_size_mutating(place_group& group)
{
  const size_t n = (1u << 20) + 7;
  auto a         = sharded_array<long long>::allocate(group, n);
  sequence(a, default_envs(a), 0LL, 1LL);
  barrier(default_envs(a));

  auto v = split_in_halves(a);

  // Geometry BEFORE the size-mutating verb: select_if commits the kept sizes
  // and re-tiles the global offsets, so expectations must come from here.
  struct geom
  {
    long long* data;
    size_t offset;
    size_t size;
  };
  ::std::vector<geom> before;
  for (size_t g = 0; g < v.num_shards(); ++g)
  {
    const auto& s = v.shard(g);
    before.push_back({s.data, s.global_offset, s.size});
  }

  bool threw  = false;
  size_t kept = 0;
  try
  {
    kept = select_if(v, is_even{});
  }
  catch (const ::std::exception& e)
  {
    threw = true;
    printf("  select_if over 2P view refused: %s\n", e.what());
  }
  if (!threw)
  {
    // Every half-shard keeps its even elements, in order, in place.
    size_t expected_kept = 0;
    for (size_t g = 0; g < v.num_shards(); ++g)
    {
      const geom& b         = before[g];
      const long long first = static_cast<long long>(b.offset);
      const long long count = static_cast<long long>(b.size);
      const long long evens = (first + count + 1) / 2 - (first + 1) / 2; // evens in [first, first+count)
      expected_kept += static_cast<size_t>(evens);
      EXPECT(v.shard(g).size == static_cast<size_t>(evens));
      EXPECT(v.shard(g).data == b.data);
      ::std::vector<long long> h(static_cast<size_t>(evens));
      cuda_safe_call(cudaMemcpy(h.data(), b.data, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
      long long x = (first % 2 == 0) ? first : first + 1;
      for (long long k = 0; k < evens; ++k, x += 2)
      {
        EXPECT(h[static_cast<size_t>(k)] == x);
      }
    }
    EXPECT(kept == expected_kept);
    printf("  select_if over %zu shards kept %zu of %zu: PASSED\n", v.num_shards(), kept, n);
  }
}

} // namespace

int main()
{
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda_safe_call(cudaSetDevice(0));
  auto group = place_group{exec_place::all_locality_domains()};
  printf("repeated_places: %zu places\n", group.size());

  test_map_and_combine(group);
  test_size_mutating(group);

  printf("repeated_places: PASSED\n");
  return 0;
}
