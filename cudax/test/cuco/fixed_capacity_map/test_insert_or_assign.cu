// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__cccl_config>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/iterator>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map_ref.cuh>

#include <cooperative_groups.h>
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API Pair operator()(int i) const noexcept
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

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int i) const noexcept
  {
    const auto value = values[i];
    return duplicates ? (value >= i + offset && value < count + offset && (value - offset) % 17 == i)
                      : value == i + offset;
  }
};

constexpr int block_size = 128;

template <class Ref, class Iterator>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(block_size) void device_insert_or_assign(Ref ref, Iterator first, int n)
{
  const auto group = cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());
  const auto i     = (cuda::std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x) / Ref::cg_size;
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
  SECTION("assign values initially inserted with insert")
  {
    REQUIRE(map.insert(stream, pairs(7), pairs(7) + num_keys) == num_keys);
    map.insert_or_assign(stream, pairs(19), pairs(19) + num_keys);
    verify(num_keys, 19);
  }
  SECTION("cooperative device insertion and replacement")
  {
    const auto run = [&](int offset) {
      const auto config = ::cuda::make_config(
        ::cuda::grid_dims(static_cast<unsigned>((num_keys * cg_size + block_size - 1) / block_size)),
        ::cuda::block_dims<block_size>());
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

struct constant_hash
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(int) const noexcept
  {
    return 0;
  }
};

struct erased_assign
{
  template <class Ref>
  _CCCL_DEVICE_API void operator()(Ref ref, int* result, bool existing_key) const
  {
    const auto group = cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());
    const auto slots = ref.storage_span();
    if (group.thread_rank() == 0 && existing_key)
    {
      // Hash zero starts at slot zero. The key must be found past the erased slots.
      slots.back() = typename Ref::value_type{0, 1};
    }
    group.sync();
    if constexpr (Ref::cg_size == 1)
    {
      ref.insert_or_assign(typename Ref::value_type{0, 7});
    }
    else
    {
      ref.insert_or_assign(group, typename Ref::value_type{0, 7});
    }
    group.sync();
    const auto found = ref.find(group, 0);
    if (group.thread_rank() == 0)
    {
      result[0] = found == ref.end() ? -1 : found->second;
    }
    group.sync();
    ref.insert_or_assign(group, typename Ref::value_type{0, 19});
    group.sync();
    if (group.thread_rank() == 0)
    {
      int count = 0;
      int value = -1;
      for (const auto& slot : slots)
      {
        if (slot.first == 0)
        {
          ++count;
          value = slot.second;
        }
      }
      result[1] = count == 1 ? value : -1;
    }
  }
};

using scopes  = c2h::type_list<cuda::std::integral_constant<cuda::thread_scope, cuda::thread_scope_device>,
                               cuda::std::integral_constant<cuda::thread_scope, cuda::thread_scope_block>>;
using extents = c2h::type_list<cuda::std::false_type, cuda::std::true_type>;

C2H_TEST("fixed_capacity_map insert_or_assign erased slots", "[container][erased]", cg_sizes, scopes, extents)
{
  [[maybe_unused]] constexpr int cg_size        = c2h::get<0, TestType>::value;
  [[maybe_unused]] constexpr auto scope         = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr bool static_extent = c2h::get<2, TestType>::value;
  using probing                                 = cudax::cuco::linear_probing<cg_size, constant_hash>;
  constexpr auto capacity                       = cudax::cuco::make_valid_capacity<probing, 1>(8);
  using ref_type                                = cudax::cuco::fixed_capacity_map_ref<
    int,
    int,
    scope,
    cuda::std::equal_to<int>,
    probing,
    1,
    static_extent ? capacity : cuda::std::dynamic_extent>;
  using pair_type = typename ref_type::value_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr     = cuda::device_default_memory_pool(stream.device());
  auto slots  = cuda::make_buffer<pair_type>(stream, mr, capacity, pair_type{-2, -1});
  auto result = cuda::make_buffer<int>(stream, mr, 2, -1);
  const ref_type ref{
    cudax::cuco::empty_key{-1},
    cudax::cuco::empty_value{-1},
    cudax::cuco::erased_key{-2},
    {},
    {},
    typename ref_type::storage_span_type{slots.data(), capacity}};
  const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, mr);
  const auto run    = [&](bool existing_key) {
    cuda::launch(
      stream,
      cuda::make_config(cuda::grid_dims(1), cuda::block_dims<cg_size>()),
      erased_assign{},
      ref,
      result.data(),
      existing_key);
    REQUIRE(cuda::std::all_of(
      policy,
      cuda::counting_iterator<int>{0},
      cuda::counting_iterator<int>{1},
      matches_payload<int>{result.data(), 7, 1, false}));
    REQUIRE(cuda::std::all_of(
      policy,
      cuda::counting_iterator<int>{0},
      cuda::counting_iterator<int>{1},
      matches_payload<int>{result.data() + 1, 19, 1, false}));
  };
  SECTION("reuse erased storage")
  {
    run(false);
  }
  SECTION("assign past erased slots without creating a duplicate")
  {
    run(true);
  }
}

C2H_TEST("fixed_capacity_map insert_or_assign contended erased slots", "[container][erased]", cg_sizes)
{
  [[maybe_unused]] constexpr int cg_size = c2h::get<0, TestType>::value;
  using probing                          = cudax::cuco::linear_probing<cg_size, constant_hash>;
  using map_type                         = cudax::cuco::
    fixed_capacity_map<int, int, cuda::std::dynamic_extent, cuda::thread_scope_device, cuda::std::equal_to<int>, probing>;
  using pair_type = typename map_type::value_type;
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr           = cuda::device_default_memory_pool(stream.device());
  const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, mr);
  map_type map{
    stream,
    mr,
    cuda::std::size_t{64},
    cudax::cuco::empty_key{-1},
    cudax::cuco::empty_value{-1},
    cudax::cuco::erased_key{-2}};
  cuda::std::fill(policy, map.data(), map.data() + map.capacity(), pair_type{-2, -1});
  const auto pairs = cuda::transform_iterator{cuda::counting_iterator<int>{0}, make_pair<pair_type>{7, 17}};
  map.insert_or_assign_async(stream, pairs, pairs + 400);
  REQUIRE(map.size(stream) == 17);
  auto results = cuda::make_buffer<int>(stream, mr, 17, -1);
  map.find(stream, cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{17}, results.begin());
  REQUIRE(cuda::std::all_of(
    policy,
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{17},
    matches_payload<int>{results.data(), 7, 400, true}));
}
