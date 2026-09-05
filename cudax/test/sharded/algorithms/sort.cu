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
 * @brief Correctness of `sharded::sort` (shared-address-space engine)
 *        against `std::sort` over mixed distributions, uneven shards, custom
 *        comparators and the contiguous backing; shard metadata invariants;
 *        run-to-run reproducibility.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
template <typename T, typename HostCompare>
void check_sorts_like_std(place_group& group, sharded_array<T>& data, ::std::vector<T> host, HostCompare hcomp)
{
  // Record the shard layout: sort must not change it.
  ::std::vector<size_t> sizes_before, offsets_before, caps_before;
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    sizes_before.push_back(data.shard(g).size);
    offsets_before.push_back(data.shard(g).global_offset);
    caps_before.push_back(data.shard(g).capacity);
  }
  const size_t total_before = data.size();

  data.copy_from_host(host.data());
  sort(group, data);

  ::std::vector<T> got(host.size());
  data.copy_to_host(got.data());

  ::std::sort(host.begin(), host.end(), hcomp);
  EXPECT(::std::memcmp(got.data(), host.data(), host.size() * sizeof(T)) == 0);

  // Sizes, offsets, capacities and the total are preserved.
  EXPECT(data.size() == total_before);
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    EXPECT(data.shard(g).size == sizes_before[g]);
    EXPECT(data.shard(g).global_offset == offsets_before[g]);
    EXPECT(data.shard(g).capacity == caps_before[g]);
  }
  EXPECT(data.validate());
}

void test_distributions(place_group& group)
{
  const size_t n = (1 << 20) + 37; // odd total: uneven remainder distribution

  // Uneven shards: first shard gets ~2x the elements of the others.
  ::std::vector<size_t> sizes(group.size(), n / (2 * group.size() - 1));
  sizes[0] = n - (group.size() - 1) * (n / (2 * group.size() - 1));

  auto data = sharded_array<float>::allocate(group, sizes);
  ::std::vector<float> host(n);

  // uniform random
  {
    ::std::mt19937 rng(123);
    ::std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : host)
    {
      v = dist(rng);
    }
    check_sorts_like_std(group, data, host, ::std::less<float>{});
  }

  // all-equal
  {
    ::std::fill(host.begin(), host.end(), 42.0f);
    check_sorts_like_std(group, data, host, ::std::less<float>{});
  }

  // pre-sorted
  {
    for (size_t i = 0; i < n; i++)
    {
      host[i] = static_cast<float>(i);
    }
    check_sorts_like_std(group, data, host, ::std::less<float>{});
  }

  // reverse-sorted
  {
    for (size_t i = 0; i < n; i++)
    {
      host[i] = static_cast<float>(n - i);
    }
    check_sorts_like_std(group, data, host, ::std::less<float>{});
  }

  // integer keys with heavy duplication (few distinct values)
  {
    auto idata = sharded_array<int>::allocate(group, sizes);
    ::std::vector<int> ihost(n);
    ::std::mt19937 rng(7);
    ::std::uniform_int_distribution<int> dist(0, 15);
    for (auto& v : ihost)
    {
      v = dist(rng);
    }
    check_sorts_like_std(group, idata, ihost, ::std::less<int>{});
  }
}

struct descending
{
  __host__ __device__ bool operator()(int a, int b) const
  {
    return a > b;
  }
};

void test_custom_comparator(place_group& group)
{
  const size_t n = 300001;
  auto data      = sharded_array<int>::allocate(group, n);
  ::std::vector<int> host(n);
  ::std::mt19937 rng(99);
  ::std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (auto& v : host)
  {
    v = dist(rng);
  }

  data.copy_from_host(host.data());
  sort(group, data, descending{});

  ::std::vector<int> got(n);
  data.copy_to_host(got.data());
  ::std::sort(host.begin(), host.end(), ::std::greater<int>{});
  EXPECT(::std::memcmp(got.data(), host.data(), n * sizeof(int)) == 0);
}

void test_contiguous(place_group& group)
{
  // Sorting a contiguous array must leave contiguous_data() reading as ONE
  // globally sorted array: the engine redistributes each rank's slice back to
  // the shard's original count, which IS the fixed boundary of the backing.
  const size_t n = 500000;
  auto data      = sharded_array<long long>::allocate_contiguous(group, n);
  EXPECT(data.is_contiguous());

  ::std::vector<long long> host(n);
  ::std::mt19937_64 rng(2026);
  for (auto& v : host)
  {
    v = static_cast<long long>(rng());
  }
  data.copy_from_host(host.data());

  sort(group, data);
  EXPECT(data.validate());

  // Read the whole array THROUGH THE BASE POINTER, as one plain array.
  ::std::vector<long long> got(n);
  cuda_safe_call(cudaMemcpy(got.data(), data.contiguous_data(), n * sizeof(long long), cudaMemcpyDefault));

  ::std::sort(host.begin(), host.end());
  EXPECT(::std::memcmp(got.data(), host.data(), n * sizeof(long long)) == 0);
}

void test_repeated_runs(place_group& group)
{
  // Keys-only sort: the sorted sequence is unique as a multiset, so repeated
  // runs on the same input must be byte-identical whatever the engine's
  // internal choices. Checked empirically over 3 runs.
  const size_t n = 250007;
  auto data      = sharded_array<float>::allocate(group, n);
  ::std::vector<float> host(n);
  ::std::mt19937 rng(5);
  ::std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& v : host)
  {
    v = dist(rng);
  }

  ::std::vector<float> first(n), again(n);
  data.copy_from_host(host.data());
  sort(group, data);
  data.copy_to_host(first.data());

  for (int rep = 0; rep < 2; rep++)
  {
    data.copy_from_host(host.data());
    sort(group, data);
    data.copy_to_host(again.data());
    EXPECT(::std::memcmp(again.data(), first.data(), n * sizeof(float)) == 0);
  }
}

void test_shape_mismatch(place_group& group)
{
  // An array whose shard count differs from the group's place count is
  // refused before any engine work starts.
  ::std::vector<::std::pair<size_t, data_place>> specs;
  for (size_t i = 0; i < group.size() + 1; i++)
  {
    specs.emplace_back(64, group.place(i % group.size()).affine_data_place());
  }
  auto bad = sharded_array<int>::allocate(specs);

  bool threw = false;
  try
  {
    sort(group, bad);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  // Empty arrays are a no-op (like the other sharded collectives).
  sharded_array<int> empty;
  sort(group, empty);
  EXPECT(empty.size() == 0);
}

// The P > 2 selection path (sample, then window, both exact) has its own
// edge cases: ties across runs, descending/custom comparators, uneven and
// empty shards, inputs smaller than the sample stride. Exercise it through
// the public API on ONE device by grouping several copies of one place: the
// engine only requires a shared address space, not distinct places.
template <typename T, typename Cmp, typename HostCmp>
void run_case(place_group& group, const ::std::vector<size_t>& sizes, ::std::vector<T> host, Cmp cmp, HostCmp hcmp)
{
  size_t n = 0;
  for (auto v : sizes)
  {
    n += v;
  }
  EXPECT(host.size() == n);
  auto data = sharded_array<T>::allocate(group, sizes);
  data.copy_from_host(host.data());
  sort(group, data, cmp);
  ::std::vector<T> got(n);
  data.copy_to_host(got.data());
  ::std::sort(host.begin(), host.end(), hcmp);
  EXPECT(::std::memcmp(got.data(), host.data(), n * sizeof(T)) == 0);
  EXPECT(data.validate());
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    EXPECT(data.shard(g).size == sizes[g]);
  }
}

void test_many_places(const cuda::experimental::places::exec_place& place)
{
  ::std::mt19937_64 rng(77);
  for (size_t P : {size_t{3}, size_t{4}, size_t{5}, size_t{7}})
  {
    place_group group{::std::vector<cuda::experimental::places::exec_place>(P, place)};
    auto even = [&](size_t n) {
      ::std::vector<size_t> sz(P, n / P);
      sz[P - 1] += n % P;
      return sz;
    };
    auto rnd = [&](size_t n, auto gen) {
      ::std::vector<decltype(gen())> h(n);
      for (auto& v : h)
      {
        v = gen();
      }
      return h;
    };

    // Random, even split, larger than the sample stride.
    {
      const size_t n = (size_t{1} << 20) + 37;
      run_case<int>(
        group,
        even(n),
        rnd(n,
            [&] {
              return static_cast<int>(rng());
            }),
        ::cuda::std::less<int>{},
        ::std::less<int>{});
    }
    // Heavy ties across runs (16 distinct keys), descending custom comparator (merge-sort path).
    {
      const size_t n = 400009;
      run_case<int>(
        group,
        even(n),
        rnd(n,
            [&] {
              return static_cast<int>(rng() % 16);
            }),
        descending{},
        ::std::greater<int>{});
    }
    // All keys equal.
    {
      const size_t n = 200003;
      run_case<int>(group, even(n), ::std::vector<int>(n, 42), ::cuda::std::less<int>{}, ::std::less<int>{});
    }
    // Uneven sizes with empty shards, at the ends and inside.
    {
      ::std::vector<size_t> sz(P, 0);
      sz[1] = 100003;
      if (P > 3)
      {
        sz[P - 2] = 77777;
      }
      size_t n = 0;
      for (auto v : sz)
      {
        n += v;
      }
      run_case<long long>(
        group,
        sz,
        rnd(n,
            [&] {
              return static_cast<long long>(rng());
            }),
        ::cuda::std::less<long long>{},
        ::std::less<long long>{});
    }
    // Tiny inputs: one key per shard, fewer keys than shards, a single key.
    {
      ::std::vector<size_t> one(P, 1);
      run_case<int>(
        group,
        one,
        rnd(P,
            [&] {
              return static_cast<int>(rng() % 5);
            }),
        ::cuda::std::less<int>{},
        ::std::less<int>{});
      ::std::vector<size_t> few(P, 0);
      few[0]     = 2;
      few[P - 1] = 1;
      run_case<int>(group, few, {9, 3, 7}, ::cuda::std::less<int>{}, ::std::less<int>{});
      ::std::vector<size_t> single(P, 0);
      single[P / 2] = 1;
      run_case<int>(group, single, {5}, ::cuda::std::less<int>{}, ::std::less<int>{});
    }
    // Presorted and reverse-sorted (splits land exactly on run boundaries).
    {
      const size_t n = 300000;
      ::std::vector<int> asc(n), desc(n);
      for (size_t i = 0; i < n; i++)
      {
        asc[i]  = static_cast<int>(i);
        desc[i] = static_cast<int>(n - i);
      }
      run_case<int>(group, even(n), asc, ::cuda::std::less<int>{}, ::std::less<int>{});
      run_case<int>(group, even(n), desc, ::cuda::std::less<int>{}, ::std::less<int>{});
    }
  }
}

// True when every place of the group is backed by the same device: the
// precondition of the shared-address-space engine.
bool on_one_device(place_group& group)
{
  const int dev = device_ordinal(group.place(0).affine_data_place());
  for (size_t i = 1; i < group.size(); i++)
  {
    if (device_ordinal(group.place(i).affine_data_place()) != dev)
    {
      return false;
    }
  }
  return true;
}

// A group spanning several devices is refused up front (documented contract:
// the cross-address-space engine is a separate change), with the array untouched.
void test_cross_device_refusal(place_group& group)
{
  const size_t n = 4096;
  auto data      = sharded_array<int>::allocate(group, n);
  ::std::vector<int> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<int>(n - i);
  }
  data.copy_from_host(host.data());

  bool threw = false;
  try
  {
    sort(group, data);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  ::std::vector<int> got(n);
  data.copy_to_host(got.data());
  EXPECT(::std::memcmp(got.data(), host.data(), n * sizeof(int)) == 0);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  // The full locality-domain grid: on a multi-GPU machine it spans devices.
  auto grid_group = place_group{make_locality_domain_grid()};
  test_shape_mismatch(grid_group);

  if (!on_one_device(grid_group))
  {
    test_cross_device_refusal(grid_group);
    printf("sort: the grid spans several devices; running the engine tests on device 0 only.\n");
  }

  // The engine proper runs on one device's domains (the whole grid when that
  // is already single-device).
  auto group = on_one_device(grid_group) ? ::std::move(grid_group) : place_group{make_locality_domain_grid(0)};

  test_distributions(group);
  test_custom_comparator(group);
  test_contiguous(group);
  test_repeated_runs(group);
  test_many_places(group.place(0));

  printf("sort: all tests passed\n");
  return 0;
}
