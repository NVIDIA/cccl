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

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

template <class Pair>
struct iota_pair
{
  _CCCL_HOST_DEVICE_API Pair operator()(typename Pair::first_type key) const noexcept
  {
    return Pair{key, key};
  }
};

struct is_even
{
  _CCCL_DEVICE_API bool operator()(int value) const noexcept
  {
    return value % 2 == 0;
  }
};

struct is_odd
{
  _CCCL_DEVICE_API bool operator()(int value) const noexcept
  {
    return value % 2 != 0;
  }
};

struct matches_even_present_keys
{
  const int* results;
  int num_present;

  _CCCL_DEVICE_API bool operator()(int index) const noexcept
  {
    return static_cast<bool>(results[index]) == (index < num_present && index % 2 == 0);
  }
};

struct matches_odd_keys
{
  const int* results;

  _CCCL_DEVICE_API bool operator()(int index) const noexcept
  {
    return static_cast<bool>(results[index]) == (index % 2 != 0);
  }
};

C2H_TEST("fixed_capacity_map contains_if", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type            = c2h::get<0, TestType>;
  constexpr int cg_size     = c2h::get<1, TestType>::value;
  constexpr int bucket_size = c2h::get<2, TestType>::value;
  constexpr int probing     = c2h::get<3, TestType>::value;
  using hasher              = ::cuda::hash<key_type>;
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
  using value_type                      = typename map_type::value_type;
  constexpr int num_present             = 400;
  constexpr int num_queries             = 2 * num_present;
  constexpr key_type empty_key_sentinel = key_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr     = ::cuda::device_default_memory_pool(stream.device());
  auto policy = ::cuda::execution::gpu.with(::cuda::get_stream, stream).with(::cuda::mr::get_memory_resource, mr);

  map_type map{stream,
               mr,
               static_cast<::cuda::std::size_t>(2 * num_present),
               cudax::cuco::empty_key{empty_key_sentinel},
               cudax::cuco::empty_value{empty_key_sentinel}};

  const auto pairs = ::cuda::transform_iterator{::cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{}};
  map.insert(stream, pairs, pairs + num_present);

  auto results = ::cuda::make_buffer<int>(stream, mr, num_queries, 1);
  map.contains_if(
    stream,
    ::cuda::counting_iterator<key_type>{0},
    ::cuda::counting_iterator<key_type>{num_queries},
    ::cuda::counting_iterator<int>{0},
    is_even{},
    results.begin());

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_queries},
    matches_even_present_keys{results.data(), num_present}));

  map.contains_if_async(
    stream,
    ::cuda::counting_iterator<key_type>{0},
    ::cuda::counting_iterator<key_type>{num_present},
    ::cuda::counting_iterator<int>{0},
    is_odd{},
    results.begin());

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_present},
    matches_odd_keys{results.data()}));

  map.contains_if(
    stream,
    ::cuda::counting_iterator<key_type>{0},
    ::cuda::counting_iterator<key_type>{0},
    ::cuda::counting_iterator<int>{0},
    is_even{},
    results.begin());
}
