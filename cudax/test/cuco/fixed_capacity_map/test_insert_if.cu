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
  [[nodiscard]] _CCCL_HOST_DEVICE_API Pair operator()(typename Pair::first_type key) const noexcept
  {
    return Pair{key, key};
  }
};

struct is_even
{
  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int value) const noexcept
  {
    return value % 2 == 0;
  }
};

struct is_odd
{
  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int value) const noexcept
  {
    return value % 2 != 0;
  }
};

struct matches_inserted_parity
{
  const int* results;
  bool even;

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int index) const noexcept
  {
    const bool expected = (index % 2 == 0) == even;
    return static_cast<bool>(results[index]) == expected;
  }
};

C2H_TEST("fixed_capacity_map insert_if", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<3, TestType>::value;
  using hasher                               = ::cuda::hash<key_type>;
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
  constexpr int num_keys                = 400;
  constexpr key_type empty_key_sentinel = key_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr           = ::cuda::device_default_memory_pool(stream.device());
  const auto policy = ::cuda::execution::gpu.with(::cuda::get_stream, stream).with(::cuda::mr::get_memory_resource, mr);

  map_type map{stream,
               mr,
               ::cuda::std::size_t{num_keys} * 2,
               cudax::cuco::empty_key{empty_key_sentinel},
               cudax::cuco::empty_value{empty_key_sentinel}};

  const auto pairs   = ::cuda::transform_iterator{::cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{}};
  const auto stencil = ::cuda::counting_iterator<int>{0};

  REQUIRE(map.insert_if(stream, pairs, pairs, stencil, is_even{}) == 0);
  REQUIRE(map.insert_if(stream, pairs, pairs + num_keys, stencil, is_even{}) == num_keys / 2);
  REQUIRE(map.insert_if(stream, pairs, pairs + num_keys, stencil, is_even{}) == 0);

  auto results = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.contains(
    stream, ::cuda::counting_iterator<key_type>{0}, ::cuda::counting_iterator<key_type>{num_keys}, results.begin());

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_keys},
    matches_inserted_parity{results.data(), true}));

  map.clear(stream);
  map.insert_if_async(stream, pairs, pairs + num_keys, stencil, is_odd{});
  map.contains(
    stream, ::cuda::counting_iterator<key_type>{0}, ::cuda::counting_iterator<key_type>{num_keys}, results.begin());

  REQUIRE(::cuda::std::all_of(
    policy,
    ::cuda::counting_iterator<int>{0},
    ::cuda::counting_iterator<int>{num_keys},
    matches_inserted_parity{results.data(), false}));
}
