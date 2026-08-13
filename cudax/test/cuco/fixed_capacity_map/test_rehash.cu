//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround for a cuda::buffer destructor conflict
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif // defined(__CUDACC__)

#include <cuda/__cccl_config>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cuda_runtime_api.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

template <class Map, class = void>
struct has_capacity_rehashes : ::cuda::std::false_type
{};

template <class Map>
struct has_capacity_rehashes<
  Map,
  ::cuda::std::void_t<decltype(::cuda::std::declval<Map&>().rehash(
                        ::cuda::std::declval<::cuda::stream_ref>(), ::cuda::std::declval<typename Map::size_type>())),
                      decltype(::cuda::std::declval<Map&>().rehash_async(
                        ::cuda::std::declval<::cuda::stream_ref>(), ::cuda::std::declval<typename Map::size_type>()))>>
    : ::cuda::std::true_type
{};

template <class Map, class = void>
struct has_explicit_dynamic_capacity_rehashes : ::cuda::std::false_type
{};

template <class Map>
struct has_explicit_dynamic_capacity_rehashes<
  Map,
  ::cuda::std::void_t<decltype(::cuda::std::declval<Map&>().template rehash<::cuda::std::dynamic_extent>(
                        ::cuda::std::declval<::cuda::stream_ref>(), ::cuda::std::declval<typename Map::size_type>())),
                      decltype(::cuda::std::declval<Map&>().template rehash_async<::cuda::std::dynamic_extent>(
                        ::cuda::std::declval<::cuda::stream_ref>(), ::cuda::std::declval<typename Map::size_type>()))>>
    : ::cuda::std::true_type
{};

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using mapped_types  = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

constexpr int payload_offset = 7;
constexpr int erase_modulus  = 4;

template <class Pair>
struct iota_pair
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API Pair operator()(int i) const noexcept
  {
    return Pair{static_cast<typename Pair::first_type>(i), static_cast<typename Pair::second_type>(i + payload_offset)};
  }
};

template <class Pair>
_CCCL_KERNEL_ATTRIBUTES void mark_erased_slots(
  Pair* slots,
  ::cuda::std::size_t capacity,
  typename Pair::first_type empty_key_sentinel,
  typename Pair::first_type erased_key_sentinel)
{
  using key_type = typename Pair::first_type;
  const auto idx = static_cast<::cuda::std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < capacity && slots[idx].first != empty_key_sentinel && slots[idx].first != erased_key_sentinel
      && slots[idx].first % key_type{erase_modulus} == key_type{0})
  {
    slots[idx].first = erased_key_sentinel;
  }
}

template <class Map>
_CCCL_HOST_API void mark_erased_slots(Map& map, ::cuda::stream_ref stream)
{
  constexpr int block_size = 128;
  const auto grid_size     = static_cast<unsigned>((map.capacity() + block_size - 1) / block_size);

  mark_erased_slots<<<grid_size, block_size, 0, stream.get()>>>(
    map.data(), map.capacity(), map.empty_key_sentinel(), map.erased_key_sentinel());
  REQUIRE(::cudaGetLastError() == ::cudaSuccess);
}

template <class Key, class Mapped>
struct matches_rehashed_values
{
  const Mapped* found;
  Mapped empty_value_sentinel;

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int i) const noexcept
  {
    const auto expected =
      (i % erase_modulus == 0)
        ? empty_value_sentinel
        : static_cast<Mapped>(static_cast<Key>(i) + static_cast<Key>(payload_offset));
    return found[i] == expected;
  }
};

template <class Map, class MemoryResource>
_CCCL_HOST_API void require_rehashed_contents(const Map& map, ::cuda::stream_ref stream, MemoryResource mr, int num_keys)
{
  using key_type    = typename Map::key_type;
  using mapped_type = typename Map::mapped_type;

  auto found = ::cuda::make_buffer<mapped_type>(stream, mr, num_keys, mapped_type{0});
  map.find(stream, ::cuda::counting_iterator<key_type>{0}, ::cuda::counting_iterator<key_type>{num_keys}, found.begin());

  const auto policy = ::cuda::execution::gpu.with(::cuda::get_stream, stream).with(::cuda::mr::get_memory_resource, mr);

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_keys},
    matches_rehashed_values<key_type, mapped_type>{found.data(), map.empty_value_sentinel()}));
}

C2H_TEST("fixed_capacity_map rehash", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  using mapped_type                          = c2h::get<1, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<3, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<4, TestType>::value;

  using hasher = cuda::hash<key_type>;
  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, hasher>,
                               cudax::cuco::double_hashing<cg_size, hasher>>;
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;
  using value_type = typename map_type::value_type;

  static_assert(has_capacity_rehashes<map_type>::value);
  static_assert(has_explicit_dynamic_capacity_rehashes<map_type>::value);

  constexpr int num_keys                  = 400;
  constexpr key_type empty_key_sentinel   = key_type{-1};
  constexpr key_type erased_key_sentinel  = key_type{-2};
  constexpr mapped_type empty_value       = mapped_type{-1};
  constexpr ::cuda::std::size_t requested = num_keys * 2;

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr    = ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  const auto pairs = ::cuda::transform_iterator(::cuda::counting_iterator<int>{0}, iota_pair<value_type>{});

  map_type map{
    stream,
    mr,
    requested,
    cudax::cuco::empty_key{empty_key_sentinel},
    cudax::cuco::empty_value{empty_value},
    cudax::cuco::erased_key{erased_key_sentinel}};

  REQUIRE(map.insert(stream, pairs, pairs + num_keys) == num_keys);
  mark_erased_slots(map, stream);

  const auto initial_capacity = map.capacity();
  map.rehash(stream);
  REQUIRE(map.capacity() == initial_capacity);
  require_rehashed_contents(map, stream, mr, num_keys);

  map.rehash_async(stream);
  REQUIRE(map.capacity() == initial_capacity);
  require_rehashed_contents(map, stream, mr, num_keys);

  const auto resize_request   = initial_capacity * 2;
  const auto resized_capacity = cudax::cuco::make_valid_capacity<probing_type, bucket_size>(resize_request);
  map.rehash(stream, resize_request);
  REQUIRE(map.capacity() == resized_capacity);
  require_rehashed_contents(map, stream, mr, num_keys);

  const auto async_resize_request   = resized_capacity * 2;
  const auto async_resized_capacity = cudax::cuco::make_valid_capacity<probing_type, bucket_size>(async_resize_request);
  map.rehash_async(stream, async_resize_request);
  REQUIRE(map.capacity() == async_resized_capacity);
  require_rehashed_contents(map, stream, mr, num_keys);

  constexpr auto static_capacity = cudax::cuco::make_valid_capacity<probing_type, bucket_size>(requested);
  using static_map_type          = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    static_capacity,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;

  static_assert(!has_capacity_rehashes<static_map_type>::value);
  static_assert(!has_explicit_dynamic_capacity_rehashes<static_map_type>::value);

  static_map_type static_map{
    stream,
    mr,
    cudax::cuco::empty_key{empty_key_sentinel},
    cudax::cuco::empty_value{empty_value},
    cudax::cuco::erased_key{erased_key_sentinel}};

  REQUIRE(static_map.insert(stream, pairs, pairs + num_keys) == num_keys);
  mark_erased_slots(static_map, stream);

  static_map.rehash(stream);
  REQUIRE(static_map.capacity() == static_capacity);
  require_rehashed_contents(static_map, stream, mr, num_keys);

  static_map.rehash_async(stream);
  REQUIRE(static_map.capacity() == static_capacity);
  require_rehashed_contents(static_map, stream, mr, num_keys);
}
