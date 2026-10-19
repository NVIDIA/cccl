// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host-bulk vs device-ref equivalence. The host-bulk APIs route through `add_n` /
// `contains_n` in `kernels.cuh`; the device-ref APIs route through scalar / CG / CG-range
// methods on `bloom_filter_ref` directly. Both paths share `policy_.array_pattern(...)`,
// so given the same input keys both must produce byte-identical filter bitsets (add side)
// and byte-identical query results (contains side). Catches regressions like the
// `add_coop(group, first, idx, is_valid)` iterator-formation UB fix where one path
// silently diverged from the other.

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/functional>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>

#include <cuda/experimental/__cuco/bloom_filter.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

#include <c2h/vector.h>
using cuda::std::int32_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

namespace cg   = cooperative_groups;
namespace cuco = cuda::experimental::cuco;

using size_type = int32_t;

template <class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void scalar_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
  {
    ref.add(keys[i]);
  }
}

template <int CGSize, class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void cg_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx < n)
  {
    ref.add(tile, keys[idx]);
  }
}

template <int CGSize, class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void cg_range_add_kernel(Ref ref, Key const* first, Key const* last)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.add(tile, first, last);
}

template <class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void scalar_contains_kernel(Ref ref, Key const* keys, size_type n, bool* out)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = ref.contains(keys[i]);
  }
}

template <int CGSize, class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void cg_contains_kernel(Ref ref, Key const* keys, size_type n, bool* out)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx < n)
  {
    auto const found = ref.contains(tile, keys[idx]);
    if (tile.thread_rank() == 0)
    {
      out[idx] = found;
    }
  }
}

template <int CGSize, class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void cg_range_contains_kernel(Ref ref, Key const* first, Key const* last, bool* out)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.contains(tile, first, last, out);
}

template <class Key, class Policy>
struct config_0
{
  using key_type    = Key;
  using policy_type = Policy;
};

using cases_0 = c2h::type_list<
  config_0<int32_t, cuco::bloom_filter_policy<int32_t>>,
  config_0<
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 1, 1, 1, 1, 1, 1>>,
  config_0<
    uint64_t,
    cuco::
      bloom_filter_policy<uint64_t, cuda::hash<uint64_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 12, 8, 1, 4, 2>>,
  config_0<
    float,
    cuco::
      bloom_filter_policy<float, cuda::hash<float, cuda::hash_algorithm::xxhash_64>, unsigned long long, 4, 4, 2, 2, 1, 2>>,
  config_0<
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 2, 2, 1, 8>>>;

C2H_TEST("bloom_filter: host bulk add equals device ref add", "[bloom_filter]", cases_0)
{
  using cfg    = c2h::get<0, TestType>;
  using Key    = typename cfg::key_type;
  using Policy = typename cfg::policy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using filter_type = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  // Prime: forces a partial boundary tile (mix of valid + invalid lanes) on every
  // CGSize > 1 path, exercising the cooperative is_valid mask logic.
  constexpr size_type num_keys = 397;

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());
  auto const keys_raw = thrust::raw_pointer_cast(keys.data());

  SECTION("scalar ref.add")
  {
    auto filter_a = filter_type{stream, mr, 1000};
    auto filter_b = filter_type{stream, mr, 1000};

    filter_a.add(stream, keys.begin(), keys.end());

    auto ref                 = filter_b.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_keys + block_size - 1) / block_size;
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
      scalar_add_kernel<decltype(ref), Key>,
      ref,
      keys_raw,
      num_keys);
    stream.sync();

    auto const total_words = static_cast<cuda::std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()), filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }

  SECTION("CG ref.add(group, key)")
  {
    constexpr int CGSize = Policy::add_horizontal_layout;
    auto filter_a        = filter_type{stream, mr, 1000};
    auto filter_b        = filter_type{stream, mr, 1000};

    filter_a.add(stream, keys.begin(), keys.end());

    auto ref                 = filter_b.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_keys * CGSize + block_size - 1) / block_size;
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
      cg_add_kernel<CGSize, decltype(ref), Key>,
      ref,
      keys_raw,
      num_keys);
    stream.sync();

    auto const total_words = static_cast<cuda::std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()), filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }

  SECTION("CG ref.add(group, first, last)")
  {
    constexpr int CGSize = Policy::add_horizontal_layout;
    auto filter_a        = filter_type{stream, mr, 1000};
    auto filter_b        = filter_type{stream, mr, 1000};

    filter_a.add(stream, keys.begin(), keys.end());

    auto ref = filter_b.ref();
    // Single tile processes the entire range cooperatively.
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(1), cuda::block_dims(CGSize)),
      cg_range_add_kernel<CGSize, decltype(ref), Key>,
      ref,
      keys_raw,
      keys_raw + num_keys);
    stream.sync();

    auto const total_words = static_cast<cuda::std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()), filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }
}

template <class Key, class Policy>
struct config_1
{
  using key_type    = Key;
  using policy_type = Policy;
};

using cases_1 = c2h::type_list<
  config_1<int32_t, cuco::bloom_filter_policy<int32_t>>,
  config_1<
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 1, 1, 1, 1, 1, 1>>,
  config_1<
    uint64_t,
    cuco::
      bloom_filter_policy<uint64_t, cuda::hash<uint64_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 12, 8, 1, 4, 2>>,
  config_1<
    float,
    cuco::
      bloom_filter_policy<float, cuda::hash<float, cuda::hash_algorithm::xxhash_64>, unsigned long long, 4, 4, 2, 2, 1, 2>>,
  config_1<
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 2, 2, 1, 8>>>;

C2H_TEST("bloom_filter: host bulk contains equals device ref contains", "[bloom_filter]", cases_1)
{
  using cfg    = c2h::get<0, TestType>;
  using Key    = typename cfg::key_type;
  using Policy = typename cfg::policy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using filter_type = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  // Primes: force partial boundary tiles on both insert and probe ranges across every
  // CGSize > 1 path.
  constexpr size_type num_keys  = 397;
  constexpr size_type num_probe = 797; // mix of inserted (first ~half) and non-inserted (rest)

  auto filter = filter_type{stream, mr, 1000};

  c2h::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), insert_keys.begin(), insert_keys.end());
  filter.add(stream, insert_keys.begin(), insert_keys.end());

  c2h::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::cuda::par.on(stream.get()), probe_keys.begin(), probe_keys.end());
  auto const probe_raw = thrust::raw_pointer_cast(probe_keys.data());

  c2h::device_vector<bool> bulk_result(num_probe);
  filter.contains(stream, probe_keys.begin(), probe_keys.end(), bulk_result.begin());

  SECTION("scalar ref.contains")
  {
    c2h::device_vector<bool> ref_result(num_probe);
    auto ref                 = filter.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_probe + block_size - 1) / block_size;
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
      scalar_contains_kernel<decltype(ref), Key>,
      ref,
      probe_raw,
      num_probe,
      thrust::raw_pointer_cast(ref_result.data()));
    stream.sync();

    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()),
      bulk_result.begin(),
      bulk_result.end(),
      ref_result.begin(),
      cuda::std::equal_to<bool>{}));
  }

  SECTION("CG ref.contains(group, key)")
  {
    constexpr int CGSize = Policy::contains_horizontal_layout;
    c2h::device_vector<bool> ref_result(num_probe);
    auto ref                 = filter.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_probe * CGSize + block_size - 1) / block_size;
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
      cg_contains_kernel<CGSize, decltype(ref), Key>,
      ref,
      probe_raw,
      num_probe,
      thrust::raw_pointer_cast(ref_result.data()));
    stream.sync();

    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()),
      bulk_result.begin(),
      bulk_result.end(),
      ref_result.begin(),
      cuda::std::equal_to<bool>{}));
  }

  SECTION("device-range CG ref.contains(group, first, last, out)")
  {
    constexpr int CGSize = Policy::contains_horizontal_layout;
    c2h::device_vector<bool> ref_result(num_probe);
    auto ref = filter.ref();
    // Single tile processes the entire range cooperatively.
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(1), cuda::block_dims(CGSize)),
      cg_range_contains_kernel<CGSize, decltype(ref), Key>,
      ref,
      probe_raw,
      probe_raw + num_probe,
      thrust::raw_pointer_cast(ref_result.data()));
    stream.sync();

    REQUIRE(thrust::equal(
      thrust::cuda::par.on(stream.get()),
      bulk_result.begin(),
      bulk_result.end(),
      ref_result.begin(),
      cuda::std::equal_to<bool>{}));
  }
}
