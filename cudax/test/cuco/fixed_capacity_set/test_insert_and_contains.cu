// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__cccl_config>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/fixed_capacity_set.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <testing.cuh>

#include "test_common.cuh"

using key_types     = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, cuda::std::int32_t, cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>;

template <class Key>
struct duplicate_key
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API Key operator()(int i) const noexcept
  {
    return static_cast<Key>(i % 17);
  }
};

C2H_TEST("fixed_capacity_set insert and contains", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                              = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size      = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size  = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing_kind = c2h::get<3, TestType>::value;
  using probing_type =
    cuda::std::conditional_t<probing_kind == 0,
                             cudax::cuco::linear_probing<cg_size, cuda::hash<key_type>>,
                             cudax::cuco::double_hashing<cg_size, cuda::hash<key_type>>>;
  using set_type = cudax::cuco::fixed_capacity_set<
    key_type,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;

  // Both present and absent query keys fit the smallest key type, with room for sentinels.
  constexpr int num_keys    = 41;
  constexpr int num_queries = 2 * num_keys;
  test_context context;
  set_type set{
    context.stream, context.mr, cuda::std::size_t{num_keys} * 2, cudax::cuco::empty_key{static_cast<key_type>(-1)}};
  const auto first = cuda::counting_iterator<key_type>{0};
  auto found       = cuda::make_buffer<int>(context.stream, context.mr, num_queries, 7);

  REQUIRE(set.insert(context.stream, first, first) == 0);
  set.insert_async(context.stream, first, first);
  set.contains(context.stream, first, first, found.begin());
  set.contains_async(context.stream, first, first, found.begin());
  REQUIRE(context.all_equal(found.data(), num_queries, 7));

  set.contains(context.stream, first, first + num_queries, found.begin());
  REQUIRE(context.matches(found.data(), num_queries, 0));
  REQUIRE(set.insert(context.stream, first, first + num_keys / 2) == num_keys / 2);
  set.insert_async(context.stream, first + num_keys / 2, first + num_keys);
  set.contains_async(context.stream, first, first + num_queries, found.begin());
  REQUIRE(context.matches(found.data(), num_queries, num_keys));
  REQUIRE(set.insert(context.stream, first, first + num_keys) == 0);
  set.contains(context.stream, first, first + num_queries, cuda::discard_iterator{});

  set.clear_async(context.stream);
  set.contains(context.stream, first, first + num_queries, found.begin());
  REQUIRE(context.matches(found.data(), num_queries, 0));

  // Duplicates within one launch must be counted once even under contention.
  const auto duplicates = cuda::transform_iterator{cuda::counting_iterator<int>{0}, duplicate_key<key_type>{}};
  REQUIRE(set.insert(context.stream, duplicates, duplicates + 400) == 17);
  set.contains(context.stream, first, first + num_queries, found.begin());
  REQUIRE(context.matches(found.data(), num_queries, 17));

  set.clear(context.stream);
  set.insert_async(context.stream, first, first + num_keys);
  set.contains(context.stream, first, first + num_queries, found.begin());
  REQUIRE(context.matches(found.data(), num_queries, num_keys));

  set_type full{context.stream, context.mr, cuda::std::size_t{13}, cudax::cuco::empty_key{static_cast<key_type>(-1)}};
  const int capacity = static_cast<int>(full.capacity());
  REQUIRE(full.insert(context.stream, first, first + capacity) == full.capacity());
  REQUIRE(full.insert(context.stream, first, first + capacity) == 0);
  REQUIRE(full.insert(context.stream, first + capacity, first + capacity + 1) == 0);
  full.insert_async(context.stream, first + capacity, first + capacity + 1);
  full.contains(context.stream, first, first + capacity + 1, found.begin());
  REQUIRE(context.matches(found.data(), capacity + 1, capacity));
}

struct insert_key
{
  int value;

  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator int() const noexcept
  {
    return value;
  }
};

// No conversion to the stored key type: lookup must use the supplied hash and equality.
struct query_key
{
  int value;
};

struct heterogeneous_hash
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(int value) const noexcept
  {
    return static_cast<cuda::std::uint32_t>(value / 2);
  }

  template <class Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(Key key) const noexcept
  {
    return operator()(key.value);
  }
};

struct heterogeneous_equal
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(int lhs, int rhs) const noexcept
  {
    return lhs / 2 == rhs / 2;
  }

  template <class Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(Key lhs, int rhs) const noexcept
  {
    return operator()(lhs.value, rhs);
  }
};

template <class Key>
struct make_heterogeneous_key
{
  int offset;

  [[nodiscard]] _CCCL_HOST_DEVICE_API Key operator()(int i) const noexcept
  {
    return Key{2 * i + offset};
  }
};

C2H_TEST("fixed_capacity_set heterogeneous keys and custom equivalence", "[container][heterogeneous]", cg_sizes)
{
  [[maybe_unused]] constexpr int cg_size = c2h::get<0, TestType>::value;
  using probing_type                     = cudax::cuco::linear_probing<cg_size, heterogeneous_hash>;
  using set_type                         = cudax::cuco::
    fixed_capacity_set<int, cuda::std::dynamic_extent, cuda::thread_scope_device, heterogeneous_equal, probing_type>;
  constexpr int num_keys = 37;
  test_context context;
  set_type set{context.stream, context.mr, cuda::std::size_t{100}, cudax::cuco::empty_key{-100}, {}, probing_type{}};
  const auto first      = cuda::counting_iterator<int>{0};
  const auto input      = cuda::transform_iterator{first, make_heterogeneous_key<insert_key>{0}};
  const auto equivalent = cuda::transform_iterator{first, make_heterogeneous_key<insert_key>{1}};
  const auto queries    = cuda::transform_iterator{first, make_heterogeneous_key<query_key>{1}};
  auto found            = cuda::make_buffer<int>(context.stream, context.mr, 2 * num_keys, -1);

  set.contains(context.stream, queries, queries + 2 * num_keys, found.begin());
  REQUIRE(context.matches(found.data(), 2 * num_keys, 0));
  REQUIRE(set.insert(context.stream, input, input + num_keys) == num_keys);
  REQUIRE(set.insert(context.stream, equivalent, equivalent + num_keys) == 0);
  set.contains_async(context.stream, queries, queries + 2 * num_keys, found.begin());
  REQUIRE(context.matches(found.data(), 2 * num_keys, num_keys));
}
