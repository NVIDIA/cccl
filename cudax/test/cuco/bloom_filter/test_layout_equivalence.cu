// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Byte-equal bitsets across (AddH, AddV) layout permutations, identical contains results
// across (ContainsH, ContainsV) layout permutations, equivalence between dynamic vs static
// block count, and invariance under the ConditionalAdd / EarlyExitContains policy knobs
// (both are optimizations that must not change results) -- all for fixed
// (Hash, Word, WordsPerBlock, PatternBits, keys).

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/functional>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cuda/experimental/__cuco/bloom_filter.cuh>

#include <testing.cuh>

#include <c2h/vector.h>
using cuda::std::int32_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

namespace cuco = cuda::experimental::cuco;

template <class AltPolicy>
struct config_0
{
  using altpolicy_type = AltPolicy;
};

using cases_0 = c2h::type_list<
  config_0<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 1, 8, 1, 8>>,
  config_0<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 2, 4, 1, 8>>,
  config_0<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 1, 8>>,
  config_0<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 2, 2, 1, 8>>,
  config_0<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 1, 1, 8>>>;

C2H_TEST("bloom_filter: bitset is invariant under (AddH, AddV) layout permutations", "[bloom_filter]", cases_0)
{
  using cfg       = c2h::get<0, TestType>;
  using AltPolicy = typename cfg::altpolicy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using Key            = int32_t;
  using default_policy = cuco::bloom_filter_policy<Key>;
  using filter_default_t =
    cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, default_policy>;
  using filter_alt_t = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, AltPolicy>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;

  auto filter_default = filter_default_t{stream, mr, num_blocks};
  auto filter_alt     = filter_alt_t{stream, mr, num_blocks};

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());

  filter_default.add(stream, keys.begin(), keys.end());
  filter_alt.add(stream, keys.begin(), keys.end());

  auto const total_words =
    static_cast<cuda::std::size_t>(filter_default.block_extent()) * filter_default_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::cuda::par.on(stream.get()), filter_default.data(), filter_default.data() + total_words, filter_alt.data()));
}

template <class AltPolicy>
struct config_1
{
  using altpolicy_type = AltPolicy;
};

using cases_1 = c2h::type_list<
  config_1<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 8, 1>>,
  config_1<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 2, 4>>,
  config_1<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 4, 2>>,
  config_1<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 2, 2>>,
  config_1<
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 1, 4>>>;

C2H_TEST("bloom_filter: contains results are invariant under (ContainsH, ContainsV) permutations",
         "[bloom_filter]",
         cases_1)
{
  using cfg       = c2h::get<0, TestType>;
  using AltPolicy = typename cfg::altpolicy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using Key            = int32_t;
  using default_policy = cuco::bloom_filter_policy<Key>;
  using filter_default_t =
    cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, default_policy>;
  using filter_alt_t = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, AltPolicy>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe  = 800; // mix of inserted and disjoint

  auto filter_default = filter_default_t{stream, mr, num_blocks};
  auto filter_alt     = filter_alt_t{stream, mr, num_blocks};

  c2h::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), insert_keys.begin(), insert_keys.end());
  filter_default.add(stream, insert_keys.begin(), insert_keys.end());
  filter_alt.add(stream, insert_keys.begin(), insert_keys.end());

  c2h::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::cuda::par.on(stream.get()), probe_keys.begin(), probe_keys.end());

  c2h::device_vector<bool> result_default(num_probe);
  c2h::device_vector<bool> result_alt(num_probe);
  filter_default.contains(stream, probe_keys.begin(), probe_keys.end(), result_default.begin());
  filter_alt.contains(stream, probe_keys.begin(), probe_keys.end(), result_alt.begin());

  REQUIRE(thrust::equal(
    thrust::cuda::par.on(stream.get()), result_default.begin(), result_default.end(), result_alt.begin()));
}

C2H_TEST("bloom_filter: bitset is invariant under dynamic vs static block count", "[bloom_filter]")
{
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using Key                              = int32_t;
  using Policy                           = cuco::bloom_filter_policy<Key>;
  constexpr cuda::std::size_t num_blocks = 1'000;
  constexpr int32_t num_keys             = 400;

  using filter_dynamic_t = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  using filter_static_t  = cuco::bloom_filter<Key, num_blocks, cuda::thread_scope_device, Policy>;

  auto filter_dynamic = filter_dynamic_t{stream, mr, num_blocks};
  auto filter_static  = filter_static_t{stream, mr};

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());

  filter_dynamic.add(stream, keys.begin(), keys.end());
  filter_static.add(stream, keys.begin(), keys.end());

  auto const total_words = num_blocks * filter_dynamic_t::words_per_block;
  REQUIRE(thrust::equal(thrust::cuda::par.on(stream.get()),
                        filter_dynamic.data(),
                        filter_dynamic.data() + total_words,
                        filter_static.data()));
}

C2H_TEST("bloom_filter: bitset is invariant under ConditionalAdd", "[bloom_filter]")
{
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using Key = int32_t;
  // Same layout, ConditionalAdd off vs on. The read-before-atomic skip must yield the same bits.
  using filter_off_t =
    cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, cuco::bloom_filter_policy<Key>>;
  using filter_on_t = cuco::bloom_filter<
    Key,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuco::bloom_filter_policy<Key,
                              cuda::hash<Key, cuda::hash_algorithm::xxhash_64>,
                              uint32_t,
                              8,
                              8,
                              8,
                              1,
                              1,
                              8,
                              cuco::conditional_add_mode::on>>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;

  auto filter_off = filter_off_t{stream, mr, num_blocks};
  auto filter_on  = filter_on_t{stream, mr, num_blocks};

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());

  // Add twice so the second pass hits already-set words, exercising the ConditionalAdd skip.
  filter_off.add(stream, keys.begin(), keys.end());
  filter_off.add(stream, keys.begin(), keys.end());
  filter_on.add(stream, keys.begin(), keys.end());
  filter_on.add(stream, keys.begin(), keys.end());

  auto const total_words = static_cast<cuda::std::size_t>(filter_off.block_extent()) * filter_off_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::cuda::par.on(stream.get()), filter_off.data(), filter_off.data() + total_words, filter_on.data()));
}

C2H_TEST("bloom_filter: contains results are invariant under EarlyExitContains", "[bloom_filter]")
{
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using Key = int32_t;
  // ContainsHorizontalLayout > 1 so the compare_patterns early-exit branch is actually used.
  using filter_off_t = cuco::bloom_filter<
    Key,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuco::bloom_filter_policy<Key, cuda::hash<Key, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 8, 1>>;
  using filter_on_t = cuco::bloom_filter<
    Key,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuco::bloom_filter_policy<Key,
                              cuda::hash<Key, cuda::hash_algorithm::xxhash_64>,
                              uint32_t,
                              8,
                              8,
                              8,
                              1,
                              8,
                              1,
                              cuco::conditional_add_mode::off,
                              cuco::early_exit_contains_mode::on>>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe  = 800; // mix of inserted and disjoint, so misses fire the early exit

  auto filter_off = filter_off_t{stream, mr, num_blocks};
  auto filter_on  = filter_on_t{stream, mr, num_blocks};

  c2h::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), insert_keys.begin(), insert_keys.end());
  filter_off.add(stream, insert_keys.begin(), insert_keys.end());
  filter_on.add(stream, insert_keys.begin(), insert_keys.end());

  c2h::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::cuda::par.on(stream.get()), probe_keys.begin(), probe_keys.end());

  c2h::device_vector<bool> result_off(num_probe);
  c2h::device_vector<bool> result_on(num_probe);
  filter_off.contains(stream, probe_keys.begin(), probe_keys.end(), result_off.begin());
  filter_on.contains(stream, probe_keys.begin(), probe_keys.end(), result_on.begin());

  REQUIRE(thrust::equal(thrust::cuda::par.on(stream.get()), result_off.begin(), result_off.end(), result_on.begin()));
}
