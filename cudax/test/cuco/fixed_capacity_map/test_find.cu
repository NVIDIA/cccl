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
#include <thrust/logical.h>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

// Payloads are offset from their key so a bug that returns the key instead of the mapped value is caught.
constexpr int payload_offset = 7;

template <class Pair>
struct iota_pair
{
  __host__ __device__ Pair operator()(typename Pair::first_type i) const noexcept
  {
    return Pair{i, static_cast<typename Pair::second_type>(i + payload_offset)};
  }
};

// Present keys [0, num_keys) find their payload (key + payload_offset) and absent keys
// [num_keys, ...) find the empty value sentinel.
template <class Key>
struct match_found
{
  const Key* found;
  int num_keys;
  Key sentinel;

  __device__ bool operator()(int i) const noexcept
  {
    return (i < num_keys) ? (found[i] == static_cast<Key>(i) + payload_offset) : (found[i] == sentinel);
  }
};

template <class Key>
struct is_not_sentinel
{
  Key sentinel;

  __device__ bool operator()(Key value) const noexcept
  {
    return value != sentinel;
  }
};

struct is_even
{
  __device__ bool operator()(int i) const noexcept
  {
    return (i % 2) == 0;
  }
};

// find_if queries only even keys; odd positions resolve to the empty value sentinel
template <class Key>
struct match_find_if
{
  const Key* found;
  Key sentinel;

  __device__ bool operator()(int i) const noexcept
  {
    return ((i % 2) == 0) ? (found[i] == static_cast<Key>(i) + payload_offset) : (found[i] == sentinel);
  }
};

C2H_TEST("fixed_capacity_map find", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<3, TestType>::value;

  using hasher = cudax::cuco::hash<key_type>;
  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, hasher>,
                               cudax::cuco::double_hashing<cg_size, hasher>>;
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    key_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys      = 400;
  constexpr key_type sentinel = key_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               static_cast<::cuda::std::size_t>(num_keys * 2),
               cudax::cuco::empty_key{sentinel},
               cudax::cuco::empty_value{sentinel}};

  auto pairs = cuda::transform_iterator(cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
  map.insert(stream, pairs, pairs + num_keys);

  // Find present keys [0, num_keys) and absent keys [num_keys, 2 * num_keys)
  auto found = ::cuda::make_buffer<key_type>(stream, mr, 2 * num_keys, key_type{0});
  map.find(stream, cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{2 * num_keys}, found.begin());

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{2 * num_keys},
    match_found<key_type>{found.data(), num_keys, sentinel}));

  // find_if only queries even keys; odd positions resolve to the empty value sentinel
  auto found_if = ::cuda::make_buffer<key_type>(stream, mr, num_keys, key_type{0});
  map.find_if(stream,
              cuda::counting_iterator<key_type>{0},
              cuda::counting_iterator<key_type>{num_keys},
              cuda::counting_iterator<int>{0},
              is_even{},
              found_if.begin());

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{num_keys},
    match_find_if<key_type>{found_if.data(), sentinel}));

  // After clear the map is empty, so every key resolves to the empty value sentinel
  map.clear(stream);
  auto cleared = ::cuda::make_buffer<key_type>(stream, mr, num_keys, key_type{0});
  map.find(stream, cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{num_keys}, cleared.begin());
  REQUIRE(::thrust::none_of(
    ::thrust::cuda::par.on(stream.get()),
    cleared.data(),
    cleared.data() + num_keys,
    is_not_sentinel<key_type>{sentinel}));
}
