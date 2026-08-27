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
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cuda_runtime_api.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using mapped_types  = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

template <class Pair>
struct duplicate_iota_pair
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API Pair operator()(int i) const noexcept
  {
    const auto key = static_cast<typename Pair::first_type>(i / 2);
    return Pair{key, static_cast<typename Pair::second_type>(key)};
  }
};

C2H_TEST("fixed_capacity_map size", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
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

  constexpr int num_inputs          = 50'000;
  constexpr int num_unique_keys     = num_inputs / 2;
  constexpr key_type empty_key      = key_type{-1};
  constexpr key_type erased_key     = key_type{-2};
  constexpr mapped_type empty_value = mapped_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{
    stream,
    mr,
    static_cast<::cuda::std::size_t>(num_inputs * 2),
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value},
    cudax::cuco::erased_key{erased_key}};

  REQUIRE(map.size(stream) == 0);

  const auto pairs = ::cuda::transform_iterator(::cuda::counting_iterator<int>{0}, duplicate_iota_pair<value_type>{});
  map.insert_async(stream, pairs, pairs + num_inputs);
  REQUIRE(map.size(stream) == num_unique_keys);

  map.insert_async(stream, pairs, pairs + num_inputs);
  REQUIRE(map.size(stream) == num_unique_keys);

  map.clear_async(stream);

  const value_type erased_slot{erased_key, mapped_type{0}};
  REQUIRE_CUDART(cudaMemcpyAsync(map.data(), &erased_slot, sizeof(value_type), cudaMemcpyHostToDevice, stream.get()));

  REQUIRE(map.size(stream) == 0);
}
