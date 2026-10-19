// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises `cuco::bloom_filter_ref` device-side APIs (`ref.add(key)`, `ref.contains(key)`,
// `ref.contains(group, key)`) directly from custom kernels, varying the cooperative group size.
// The bulk host-side APIs route through the warp-cooperative kernel which already had correct
// CG-reduction semantics; the scalar and CG ref methods are separate code paths that need their
// own coverage.

#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/functional>
#include <cuda/launch>
#include <cuda/memory_pool>
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

using size_type = int32_t;

namespace cg   = cooperative_groups;
namespace cuco = cuda::experimental::cuco;

template <class Ref, class Key>
_CCCL_KERNEL_ATTRIBUTES void scalar_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
  {
    ref.add(keys[i]);
  }
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
_CCCL_KERNEL_ATTRIBUTES void cg_contains_consistency_kernel(Ref ref, Key const* keys, size_type n, int* mismatches)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx >= n)
  {
    return;
  }
  bool const got       = ref.contains(tile, keys[idx]);
  bool const all_agree = tile.all(got);
  bool const any_agree = tile.any(got);
  if (tile.thread_rank() == 0 && (all_agree != any_agree || got != ref.contains(keys[idx])))
  {
    atomicAdd(mismatches, 1);
  }
}

template <int CGSize, class Ref>
_CCCL_KERNEL_ATTRIBUTES void cooperative_clear_kernel(Ref ref)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.clear(tile);
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
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>>;

C2H_TEST("bloom_filter device ref scalar add and contains", "[bloom_filter]", cases_0)
{
  using cfg    = c2h::get<0, TestType>;
  using Key    = typename cfg::key_type;
  using Policy = typename cfg::policy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using filter_type = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{stream, mr, 1000};

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());
  c2h::device_vector<bool> contained(num_keys, false);

  auto ref = filter.ref();

  constexpr int block_size = 128;
  int const grid_size      = (num_keys + block_size - 1) / block_size;

  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
    scalar_add_kernel<decltype(ref), Key>,
    ref,
    thrust::raw_pointer_cast(keys.data()),
    num_keys);
  stream.sync();

  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
    scalar_contains_kernel<decltype(ref), Key>,
    ref,
    thrust::raw_pointer_cast(keys.data()),
    num_keys,
    thrust::raw_pointer_cast(contained.data()));
  stream.sync();

  REQUIRE(
    thrust::all_of(thrust::cuda::par.on(stream.get()), contained.begin(), contained.end(), cuda::std::identity{}));
}

template <int32_t CGSize, class Key, class Policy>
struct config_1
{
  static constexpr int32_t cgsize_v = CGSize;
  using key_type                    = Key;
  using policy_type                 = Policy;
};

using cases_1 = c2h::type_list<
  config_1<
    1,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    2,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    8,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    16,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    32,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    4,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 4, 2, 4, 2>>,
  config_1<
    8,
    int32_t,
    cuco::bloom_filter_policy<int32_t, cuda::hash<int32_t, cuda::hash_algorithm::xxhash_64>, uint32_t, 8, 8, 8, 1, 8, 1>>>;

C2H_TEST("bloom_filter device ref CG contains is reduced across the group", "[bloom_filter]", cases_1)
{
  using cfg             = c2h::get<0, TestType>;
  constexpr auto CGSize = cfg::cgsize_v;
  using Key             = typename cfg::key_type;
  using Policy          = typename cfg::policy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using filter_type = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  // Small filter + many probes drive non-inserted keys into the partial-match regime where
  // different lanes in the tile see different per-slice match results. Without a group reduction
  // in `ref.contains(group, key)`, each lane returns its slice's partial result and the tile
  // disagrees on the answer.
  constexpr size_type num_inserted{200};
  constexpr size_type num_probed{2000};
  constexpr size_type num_blocks{16};

  auto filter = filter_type{stream, mr, num_blocks};
  auto ref    = filter.ref();

  c2h::device_vector<Key> insert_keys(num_inserted);
  thrust::sequence(thrust::cuda::par.on(stream.get()), insert_keys.begin(), insert_keys.end());
  filter.add(stream, insert_keys.begin(), insert_keys.end());

  c2h::device_vector<Key> probe_keys(num_probed);
  thrust::sequence(thrust::cuda::par.on(stream.get()), probe_keys.begin(), probe_keys.end());

  c2h::device_vector<int> mismatches(1, 0);

  constexpr int block_size = 128;
  int const grid_size      = (num_probed * CGSize + block_size - 1) / block_size;

  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims(block_size)),
    cg_contains_consistency_kernel<CGSize, decltype(ref), Key>,
    ref,
    thrust::raw_pointer_cast(probe_keys.data()),
    num_probed,
    thrust::raw_pointer_cast(mismatches.data()));
  stream.sync();

  REQUIRE(static_cast<int>(mismatches[0]) == 0);
}

template <int32_t CGSize, class Key, class Policy>
struct config_2
{
  static constexpr int32_t cgsize_v = CGSize;
  using key_type                    = Key;
  using policy_type                 = Policy;
};

using cases_2 =
  c2h::type_list<config_2<1, int32_t, cuco::bloom_filter_policy<int32_t>>,
                 config_2<4, int32_t, cuco::bloom_filter_policy<int32_t>>,
                 config_2<8, int32_t, cuco::bloom_filter_policy<int32_t>>,
                 config_2<32, int32_t, cuco::bloom_filter_policy<int32_t>>>;

C2H_TEST("bloom_filter device ref cooperative clear", "[bloom_filter]", cases_2)
{
  using cfg             = c2h::get<0, TestType>;
  constexpr auto CGSize = cfg::cgsize_v;
  using Key             = typename cfg::key_type;
  using Policy          = typename cfg::policy_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});

  using filter_type = cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{stream, mr, 1000};

  c2h::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::cuda::par.on(stream.get()), keys.begin(), keys.end());
  filter.add(stream, keys.begin(), keys.end());

  c2h::device_vector<bool> contained(num_keys, false);
  filter.contains(stream, keys.begin(), keys.end(), contained.begin());
  REQUIRE(
    thrust::all_of(thrust::cuda::par.on(stream.get()), contained.begin(), contained.end(), cuda::std::identity{}));

  // Device cooperative clear via a single tile that iterates over all filter words.
  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims(1), cuda::block_dims(CGSize)),
               cooperative_clear_kernel<CGSize, decltype(filter.ref())>,
               filter.ref());
  stream.sync();

  filter.contains(stream, keys.begin(), keys.end(), contained.begin());
  REQUIRE(
    thrust::none_of(thrust::cuda::par.on(stream.get()), contained.begin(), contained.end(), cuda::std::identity{}));
}
