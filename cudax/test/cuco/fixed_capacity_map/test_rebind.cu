//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround __host__ __device__ dtor conflict in cuda::buffer
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif // defined(__CUDACC__)

#include <cuda/__cccl_config>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;
namespace cg    = cooperative_groups;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types = c2h::type_list<::cuda::std::int8_t, ::cuda::std::int16_t, ::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

struct original_hash
{
  int seed;

  _CCCL_HOST_DEVICE_API constexpr original_hash(int seed = 0) noexcept
      : seed{seed}
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t operator()(int key) const noexcept
  {
    return static_cast<::cuda::std::size_t>(key) * 33 + static_cast<::cuda::std::size_t>(seed);
  }
};

struct offset_hash
{
  int offset;
  int seed;

  _CCCL_HOST_DEVICE_API constexpr offset_hash(int offset, int seed) noexcept
      : offset{offset}
      , seed{seed}
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t operator()(int key) const noexcept
  {
    return original_hash{seed}(key - offset);
  }
};

struct offset_equal
{
  int offset;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool operator()(int probe_key, int slot_key) const noexcept
  {
    return probe_key - offset == slot_key;
  }
};

template <class Pair>
struct iota_pair
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr Pair operator()(int key) const noexcept
  {
    return Pair{key, key};
  }
};

struct is_nonzero
{
  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int value) const noexcept
  {
    return value != 0;
  }
};

template <class Ref>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_rebound_ref(Ref ref, int offset)
{
  const auto predicate_ref = ref.rebind_key_eq(offset_equal{offset});
  if constexpr (::cuda::std::is_same_v<typename Ref::hasher, original_hash>)
  {
    return predicate_ref.rebind_hash_function(offset_hash{offset, 0});
  }
  else
  {
    return predicate_ref.rebind_hash_function(
      ::cuda::std::tuple<offset_hash, offset_hash>{offset_hash{offset, 0}, offset_hash{offset, 1}});
  }
}

template <class Ref>
__global__ void contains_with_rebound_ref(Ref ref, int offset, int num_keys, int* found)
{
  const auto tile = cg::tiled_partition<Ref::cg_size>(cg::this_thread_block());
  const int index = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / Ref::cg_size;
  if (index < num_keys)
  {
    const auto rebound_ref = make_rebound_ref(ref, offset);
    const int probe_key    = index + offset;
    bool contains;
    if constexpr (Ref::cg_size == 1)
    {
      contains = rebound_ref.contains(probe_key);
    }
    else
    {
      contains = rebound_ref.contains(tile, probe_key);
    }

    if (tile.thread_rank() == 0)
    {
      found[index] = contains;
    }
  }
}

C2H_TEST("fixed_capacity_map_ref rebind APIs", "[ref][rebind]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<3, TestType>::value;

  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, original_hash>,
                               cudax::cuco::double_hashing<cg_size, original_hash, original_hash>>;
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    key_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;

  constexpr int num_keys         = 32;
  constexpr int offset           = 64;
  constexpr key_type empty_key   = key_type{-1};
  constexpr key_type empty_value = key_type{-1};
  constexpr key_type erased_key  = key_type{-2};
  constexpr int threads          = 128;
  constexpr int keys_per_block   = threads / cg_size;

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const probing_type probing_scheme = [&] {
    if constexpr (probing == 0)
    {
      return probing_type{original_hash{0}};
    }
    else
    {
      return probing_type{original_hash{0}, original_hash{1}};
    }
  }();

  map_type map{
    stream,
    mr,
    static_cast<::cuda::std::size_t>(num_keys * 2),
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value},
    cudax::cuco::erased_key{erased_key},
    {},
    probing_scheme};

  const auto pairs =
    ::cuda::transform_iterator(::cuda::counting_iterator<int>{0}, iota_pair<typename map_type::value_type>{});
  map.insert(stream, pairs, pairs + num_keys);

  const auto ref           = map.ref();
  const auto predicate_ref = ref.rebind_key_eq(offset_equal{offset});
  const auto rebound_ref   = make_rebound_ref(ref, offset);
  using rebound_ref_type   = decltype(rebound_ref);

  static_assert(::cuda::std::is_same_v<typename rebound_ref_type::key_equal, offset_equal>);
  static_assert(rebound_ref_type::capacity_v == map_type::ref_type::capacity_v);
  static_assert(rebound_ref_type::bucket_size == map_type::ref_type::bucket_size);
  static_assert(rebound_ref_type::thread_scope == map_type::ref_type::thread_scope);

  REQUIRE(rebound_ref.capacity() == ref.capacity());
  REQUIRE(rebound_ref.empty_key_sentinel() == empty_key);
  REQUIRE(rebound_ref.empty_value_sentinel() == empty_value);
  REQUIRE(rebound_ref.erased_key_sentinel() == erased_key);
  REQUIRE(rebound_ref.storage_span().data() == ref.storage_span().data());
  REQUIRE(rebound_ref.key_eq().offset == offset);

  if constexpr (probing == 0)
  {
    REQUIRE(predicate_ref.hash_function().seed == 0);
    static_assert(::cuda::std::is_same_v<typename rebound_ref_type::hasher, offset_hash>);
    REQUIRE(rebound_ref.hash_function().offset == offset);
    REQUIRE(rebound_ref.hash_function().seed == 0);
  }
  else
  {
    REQUIRE(::cuda::std::get<0>(predicate_ref.hash_function()).seed == 0);
    REQUIRE(::cuda::std::get<1>(predicate_ref.hash_function()).seed == 1);
    using rebound_hasher = ::cuda::std::tuple<offset_hash, offset_hash>;
    static_assert(::cuda::std::is_same_v<typename rebound_ref_type::hasher, rebound_hasher>);
    REQUIRE(::cuda::std::get<0>(rebound_ref.hash_function()).offset == offset);
    REQUIRE(::cuda::std::get<1>(rebound_ref.hash_function()).offset == offset);
    REQUIRE(::cuda::std::get<0>(rebound_ref.hash_function()).seed == 0);
    REQUIRE(::cuda::std::get<1>(rebound_ref.hash_function()).seed == 1);
  }

  auto found = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  contains_with_rebound_ref<<<(num_keys + keys_per_block - 1) / keys_per_block, threads, 0, stream.get()>>>(
    ref, offset, num_keys, found.data());
  REQUIRE(cudaGetLastError() == cudaSuccess);
  const auto policy =
    ::cuda::execution::gpu.with(::cuda::get_stream, stream)
      .with(::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(::cuda::device_ref{0}));

  REQUIRE(::cuda::std::all_of(policy, found.data(), found.data() + num_keys, is_nonzero{}));
}

C2H_TEST("fixed_capacity_map_ref rebind APIs preserve static capacity", "[ref][rebind][static]")
{
  using probing_type      = cudax::cuco::linear_probing<1, original_hash>;
  constexpr auto capacity = cudax::cuco::make_valid_capacity<probing_type, 1>(::cuda::std::size_t{128});
  using map_type          = cudax::cuco::
    fixed_capacity_map<int, int, capacity, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing_type>;

  constexpr int num_keys = 32;
  constexpr int offset   = 1000;
  constexpr int threads  = 128;

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  map_type map{stream, mr, cudax::cuco::empty_key{-1}, cudax::cuco::empty_value{-1}};

  const auto pairs =
    ::cuda::transform_iterator(::cuda::counting_iterator<int>{0}, iota_pair<typename map_type::value_type>{});
  map.insert(stream, pairs, pairs + num_keys);

  const auto ref         = map.ref();
  const auto rebound_ref = make_rebound_ref(ref, offset);
  static_assert(decltype(rebound_ref)::capacity_v == capacity);
  REQUIRE(rebound_ref.capacity() == capacity);
  REQUIRE(rebound_ref.empty_key_sentinel() == -1);
  REQUIRE(rebound_ref.erased_key_sentinel() == -1);
  REQUIRE(rebound_ref.storage_span().data() == ref.storage_span().data());

  auto found = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  contains_with_rebound_ref<<<1, threads, 0, stream.get()>>>(ref, offset, num_keys, found.data());
  REQUIRE(cudaGetLastError() == cudaSuccess);
  const auto policy =
    ::cuda::execution::gpu.with(::cuda::get_stream, stream)
      .with(::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(::cuda::device_ref{0}));

  REQUIRE(::cuda::std::all_of(policy, found.data(), found.data() + num_keys, is_nonzero{}));
}
