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

#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using mapped_types  = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

constexpr int payload_offset = 7;

template <class Pair>
struct duplicate_iota_pair
{
  __host__ __device__ Pair operator()(int i) const noexcept
  {
    const auto key = static_cast<typename Pair::first_type>(i / 2);
    return Pair{key, static_cast<typename Pair::second_type>(key + payload_offset)};
  }
};

template <class Key, class Value>
struct matches_iota_pair
{
  const Key* keys;
  const Value* values;

  __device__ bool operator()(int i) const noexcept
  {
    return keys[i] == static_cast<Key>(i) && values[i] == static_cast<Value>(i) + static_cast<Value>(payload_offset);
  }
};

template <class Key>
struct matches_iota_key
{
  const Key* keys;

  __device__ bool operator()(int i) const noexcept
  {
    return keys[i] == static_cast<Key>(i);
  }
};

C2H_TEST(
  "fixed_capacity_map retrieve_all", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
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

  constexpr int num_inputs             = 500'000;
  constexpr int num_unique_keys        = num_inputs / 2;
  constexpr key_type key_sentinel      = key_type{-1};
  constexpr mapped_type value_sentinel = mapped_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               static_cast<::cuda::std::size_t>(num_inputs * 2),
               cudax::cuco::empty_key{key_sentinel},
               cudax::cuco::empty_value{value_sentinel}};

  auto keys   = ::cuda::make_buffer<key_type>(stream, mr, num_unique_keys, key_type{0});
  auto values = ::cuda::make_buffer<mapped_type>(stream, mr, num_unique_keys, mapped_type{0});

  const auto [empty_keys_end, empty_values_end] = map.retrieve_all(stream, keys.begin(), values.begin());
  REQUIRE(empty_keys_end == keys.begin());
  REQUIRE(empty_values_end == values.begin());

  const auto pairs = ::cuda::transform_iterator(::cuda::counting_iterator<int>{0}, duplicate_iota_pair<value_type>{});
  REQUIRE(map.insert(stream, pairs, pairs + num_inputs) == num_unique_keys);

  const auto [keys_end, values_end] = map.retrieve_all(stream, keys.begin(), values.begin());
  REQUIRE(keys_end == keys.begin() + num_unique_keys);
  REQUIRE(values_end == values.begin() + num_unique_keys);

  const auto policy = ::thrust::cuda::par.on(stream.get());
  ::thrust::sort_by_key(policy, keys.begin(), keys_end, values.begin());
  REQUIRE(::thrust::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_unique_keys},
    matches_iota_pair<key_type, mapped_type>{keys.data(), values.data()}));

  auto discarded_keys         = ::cuda::make_buffer<key_type>(stream, mr, num_unique_keys, key_type{0});
  const auto discarded_values = ::thrust::make_discard_iterator();
  const auto [discarded_keys_end, discarded_values_end] =
    map.retrieve_all(stream, discarded_keys.begin(), discarded_values);
  REQUIRE(discarded_keys_end == discarded_keys.begin() + num_unique_keys);
  REQUIRE(discarded_values_end == discarded_values + num_unique_keys);

  ::thrust::sort(policy, discarded_keys.begin(), discarded_keys_end);
  REQUIRE(::thrust::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_unique_keys},
    matches_iota_key<key_type>{discarded_keys.data()}));

  map.clear(stream);
  const auto [cleared_keys_end, cleared_values_end] = map.retrieve_all(stream, keys.begin(), values.begin());
  REQUIRE(cleared_keys_end == keys.begin());
  REQUIRE(cleared_values_end == values.begin());
}
