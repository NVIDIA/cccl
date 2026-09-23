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
#include <cuda/launch>
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
using mapped_types  = key_types;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

template <class Pair>
struct make_pair
{
  int offset;
  int modulus;

  _CCCL_HOST_DEVICE_API Pair operator()(int i) const noexcept
  {
    return Pair{static_cast<typename Pair::first_type>(i % modulus),
                static_cast<typename Pair::second_type>(i + offset)};
  }
};

template <class Value>
struct matches_payload
{
  const Value* values;
  int offset;
  int count;
  bool duplicates;

  _CCCL_DEVICE_API bool operator()(int i) const noexcept
  {
    const auto value = values[i];
    return duplicates ? (value >= i + offset && value < count + offset && (value - offset) % 17 == i)
                      : value == i + offset;
  }
};

template <class Ref, class Iterator>
__global__ void device_insert_or_assign(Ref ref, Iterator first, int n)
{
  const auto group = cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());
  const int i      = (blockIdx.x * blockDim.x + threadIdx.x) / Ref::cg_size;
  if (i < n)
  {
    // Exercise the cooperative overload even for cg_size == 1.
    ref.insert_or_assign(group, first[i]);
  }
}

C2H_TEST(
  "fixed_capacity_map insert_or_assign", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
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
  using value_type       = typename map_type::value_type;
  constexpr int num_keys = 400;
  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr           = ::cuda::device_default_memory_pool(stream.device());
  const auto policy = ::cuda::execution::gpu.with(::cuda::get_stream, stream).with(::cuda::mr::get_memory_resource, mr);
  map_type map{stream,
               mr,
               ::cuda::std::size_t{num_keys} * 2,
               cudax::cuco::empty_key{key_type{-1}},
               cudax::cuco::empty_value{mapped_type{-1}}};
  auto results     = ::cuda::make_buffer<mapped_type>(stream, mr, num_keys, mapped_type{-1});
  const auto keys  = ::cuda::counting_iterator<key_type>{0};
  const auto pairs = [](int offset, int modulus = 400) {
    return ::cuda::transform_iterator{::cuda::counting_iterator<int>{0}, make_pair<value_type>{offset, modulus}};
  };
  const auto verify = [&](int count, int offset, bool duplicates = false) {
    REQUIRE(map.size(stream) == static_cast<::cuda::std::size_t>(count));
    map.find(stream, keys, keys + count, results.begin());
    REQUIRE(::cuda::std::all_of(
      policy,
      ::cuda::counting_iterator<int>{0},
      ::cuda::counting_iterator<int>{count},
      matches_payload<mapped_type>{results.data(), offset, num_keys, duplicates}));
  };

  SECTION("host insertion, replacement, and mixed batches")
  {
    map.insert_or_assign(stream, pairs(7), pairs(7));
    map.insert_or_assign_async(stream, pairs(7), pairs(7));
    REQUIRE(map.size(stream) == 0);
    map.insert_or_assign(stream, pairs(7), pairs(7) + num_keys / 2);
    verify(num_keys / 2, 7);
    map.insert_or_assign_async(stream, pairs(19), pairs(19) + num_keys);
    verify(num_keys, 19);
    map.insert_or_assign(stream, pairs(31), pairs(31) + num_keys);
    verify(num_keys, 31);
  }
  SECTION("cooperative device insertion and replacement")
  {
    const auto run = [&](int offset) {
      const auto config =
        ::cuda::make_config(::cuda::grid_dims((num_keys * cg_size + 127) / 128), ::cuda::block_dims<128>());
      ::cuda::launch(
        stream,
        config,
        device_insert_or_assign<decltype(map.ref()), decltype(pairs(offset))>,
        map.ref(),
        pairs(offset),
        num_keys);
    };
    run(7);
    verify(num_keys, 7);
    run(19);
    verify(num_keys, 19);
  }
  SECTION("duplicate keys")
  {
    map.insert_or_assign_async(stream, pairs(7, 17), pairs(7, 17) + num_keys);
    verify(17, 7, true);
    map.insert_or_assign(stream, pairs(31, 17), pairs(31, 17) + num_keys);
    verify(17, 31, true);
  }
  SECTION("full table termination and assignment")
  {
    map_type full{stream,
                  mr,
                  ::cuda::std::size_t{1},
                  cudax::cuco::empty_key{key_type{-1}},
                  cudax::cuco::empty_value{mapped_type{-1}}};
    const int capacity = static_cast<int>(full.capacity());
    full.insert_or_assign(stream, pairs(7), pairs(7) + capacity);
    REQUIRE(full.size(stream) == full.capacity());
    full.insert_or_assign(stream, pairs(19), pairs(19) + capacity + 1);
    REQUIRE(full.size(stream) == full.capacity());
    full.find(stream, keys, keys + capacity, results.begin());
    REQUIRE(::cuda::std::all_of(
      policy,
      ::cuda::counting_iterator<int>{0},
      ::cuda::counting_iterator<int>{capacity},
      matches_payload<mapped_type>{results.data(), 19, capacity, false}));
  }
}
