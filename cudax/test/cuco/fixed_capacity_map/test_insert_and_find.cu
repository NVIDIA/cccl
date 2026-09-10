//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using mapped_types  = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

inline constexpr ::cuda::std::int32_t initial_payload_offset   = 7;
inline constexpr ::cuda::std::int32_t duplicate_payload_offset = 107;

template <class Pair>
struct iota_pair
{
  ::cuda::std::int32_t payload_offset;

  _CCCL_HOST_DEVICE_API Pair operator()(::cuda::std::int32_t index) const noexcept
  {
    using key_type    = typename Pair::first_type;
    using mapped_type = typename Pair::second_type;
    return Pair{static_cast<key_type>(index), static_cast<mapped_type>(index) + payload_offset};
  }
};

template <class Mapped>
struct matches_payloads
{
  const Mapped* found;
  ::cuda::std::int32_t payload_offset;

  _CCCL_DEVICE_API bool operator()(::cuda::std::int32_t index) const noexcept
  {
    return found[index] == static_cast<Mapped>(index) + payload_offset;
  }
};

struct matches_insertion_status
{
  const ::cuda::std::int32_t* inserted;
  bool expected;

  _CCCL_DEVICE_API bool operator()(::cuda::std::int32_t index) const noexcept
  {
    return static_cast<bool>(inserted[index]) == expected;
  }
};

template <class Mapped>
struct matches_device_results
{
  const Mapped* found;
  const ::cuda::std::int32_t* inserted;

  _CCCL_DEVICE_API bool operator()(::cuda::std::int32_t index) const noexcept
  {
    return found[index] == static_cast<Mapped>(initial_payload_offset)
        && static_cast<bool>(inserted[index]) == (index == 0);
  }
};

template <class Ref>
__global__ void device_insert_and_find_kernel(Ref ref, typename Ref::mapped_type* found, ::cuda::std::int32_t* inserted)
{
  using value_type  = typename Ref::value_type;
  using key_type    = typename Ref::key_type;
  using mapped_type = typename Ref::mapped_type;

  const value_type initial_value{key_type{0}, static_cast<mapped_type>(initial_payload_offset)};
  const value_type duplicate_value{key_type{0}, static_cast<mapped_type>(duplicate_payload_offset)};

  if constexpr (Ref::cg_size == 1)
  {
    if (threadIdx.x == 0)
    {
      const auto [initial_found, initial_inserted] = ref.insert_and_find(initial_value);
      found[0]                                     = initial_found->second;
      inserted[0]                                  = initial_inserted;

      const auto [duplicate_found, duplicate_inserted] = ref.insert_and_find(duplicate_value);
      found[1]                                         = duplicate_found->second;
      inserted[1]                                      = duplicate_inserted;
    }
  }
  else
  {
    const auto block = ::cooperative_groups::this_thread_block();
    const auto tile  = ::cooperative_groups::tiled_partition<Ref::cg_size, ::cooperative_groups::thread_block>(block);

    const auto [initial_found, initial_inserted] = ref.insert_and_find(tile, initial_value);
    if (tile.thread_rank() == 0)
    {
      found[0]    = initial_found->second;
      inserted[0] = initial_inserted;
    }
    tile.sync();

    const auto [duplicate_found, duplicate_inserted] = ref.insert_and_find(tile, duplicate_value);
    if (tile.thread_rank() == 0)
    {
      found[1]    = duplicate_found->second;
      inserted[1] = duplicate_inserted;
    }
  }
}

C2H_TEST(
  "fixed_capacity_map insert_and_find", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  using mapped_type                          = c2h::get<1, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<3, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<4, TestType>::value;
  using hasher                               = ::cuda::hash<key_type>;
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
  using value_type                           = typename map_type::value_type;
  using ref_type                             = typename map_type::ref_type;
  constexpr ::cuda::std::int32_t num_keys    = 400;
  constexpr key_type empty_key_sentinel      = key_type{-1};
  constexpr mapped_type empty_value_sentinel = mapped_type{-1};

  CAPTURE(sizeof(key_type), sizeof(mapped_type), cg_size, bucket_size, probing);

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr           = ::cuda::device_default_memory_pool(stream.device());
  const auto policy = ::cuda::execution::gpu.with(::cuda::get_stream, stream).with(::cuda::mr::get_memory_resource, mr);

  map_type map{stream,
               mr,
               ::cuda::std::size_t{num_keys} * 2,
               cudax::cuco::empty_key{empty_key_sentinel},
               cudax::cuco::empty_value{empty_value_sentinel}};

  const auto initial_pairs = ::cuda::transform_iterator{
    ::cuda::counting_iterator<::cuda::std::int32_t>{0}, iota_pair<value_type>{initial_payload_offset}};
  const auto duplicate_pairs = ::cuda::transform_iterator{
    ::cuda::counting_iterator<::cuda::std::int32_t>{0}, iota_pair<value_type>{duplicate_payload_offset}};

  auto found    = ::cuda::make_buffer<mapped_type>(stream, mr, num_keys, mapped_type{0});
  auto inserted = ::cuda::make_buffer<::cuda::std::int32_t>(stream, mr, num_keys, 0);

  map.insert_and_find_async(stream, initial_pairs, initial_pairs + num_keys, found.begin(), inserted.begin());

  auto queried = ::cuda::make_buffer<mapped_type>(stream, mr, num_keys, mapped_type{0});
  map.find(
    stream, ::cuda::counting_iterator<key_type>{0}, ::cuda::counting_iterator<key_type>{num_keys}, queried.begin());

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_payloads<mapped_type>{found.data(), initial_payload_offset}));
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_insertion_status{inserted.data(), true}));
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_payloads<mapped_type>{queried.data(), initial_payload_offset}));

  map.insert_and_find(stream, duplicate_pairs, duplicate_pairs + num_keys, found.begin(), inserted.begin());
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_payloads<mapped_type>{found.data(), initial_payload_offset}));
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_insertion_status{inserted.data(), false}));

  map.insert_and_find(stream, initial_pairs, initial_pairs, found.begin(), inserted.begin());
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_payloads<mapped_type>{found.data(), initial_payload_offset}));
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{num_keys},
    matches_insertion_status{inserted.data(), false}));

  map.clear(stream);
  auto device_found    = ::cuda::make_buffer<mapped_type>(stream, mr, 2, mapped_type{0});
  auto device_inserted = ::cuda::make_buffer<::cuda::std::int32_t>(stream, mr, 2, 0);

  device_insert_and_find_kernel<ref_type>
    <<<1, cg_size, 0, stream.get()>>>(map.ref(), device_found.data(), device_inserted.data());
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{2},
    matches_device_results<mapped_type>{device_found.data(), device_inserted.data()}));

  map.clear(stream);
  const auto capacity = static_cast<::cuda::std::int32_t>(map.capacity());
  REQUIRE(map.insert(stream, initial_pairs, initial_pairs + capacity) == map.capacity());
  map.insert_and_find(stream, initial_pairs + capacity, initial_pairs + capacity + 1, found.begin(), inserted.begin());
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{1},
    matches_payloads<mapped_type>{found.data(), empty_value_sentinel}));
  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<::cuda::std::int32_t>{0},
    ::cuda::counting_iterator<::cuda::std::int32_t>{1},
    matches_insertion_status{inserted.data(), false}));
}
