// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__cccl_config>
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

// Counts the occurrences of query keys that are present in the set.
C2H_TEST("fixed_capacity_set count", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
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

  constexpr int num_keys = 41;
  test_context context;
  set_type set{
    context.stream, context.mr, cuda::std::size_t{num_keys} * 2, cudax::cuco::empty_key{static_cast<key_type>(-1)}};

  const auto first = cuda::counting_iterator<key_type>{0};

  // Empty input range.
  REQUIRE(set.count(context.stream, first, first) == 0);

  // No queried keys are present.
  REQUIRE(set.count(context.stream, first, first + num_keys) == 0);

  // Insert keys [0, num_keys).
  REQUIRE(set.insert(context.stream, first, first + num_keys) == num_keys);

  // All queried keys are present.
  REQUIRE(set.count(context.stream, first, first + num_keys) == num_keys);

  // No queried keys are present.
  const auto absent = cuda::counting_iterator<key_type>{static_cast<key_type>(100)};
  REQUIRE(set.count(context.stream, absent, absent + num_keys) == 0);

  // Only the first num_keys keys in the query range are present.
  REQUIRE(set.count(context.stream, first, first + 2 * num_keys) == num_keys);

  // Duplicate query keys are counted by occurrence, not by unique key.
  const auto duplicates = cuda::transform_iterator{cuda::counting_iterator<int>{0}, duplicate_key<key_type>{}};
  REQUIRE(set.count(context.stream, duplicates, duplicates + 400) == 400);

  // Clearing the set resets the count.
  set.clear(context.stream);
  REQUIRE(set.count(context.stream, first, first + num_keys) == 0);
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
  cuda::std::uint32_t seed = 0;

  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(int value) const noexcept
  {
    return static_cast<cuda::std::uint32_t>(value / 2) ^ seed;
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

C2H_TEST(
  "fixed_capacity_set heterogeneous keys and custom equivalence", "[container][heterogeneous]", cg_sizes, probing_kinds)
{
  [[maybe_unused]] constexpr int cg_size      = c2h::get<0, TestType>::value;
  [[maybe_unused]] constexpr int probing_kind = c2h::get<1, TestType>::value;
  using probing_type =
    cuda::std::conditional_t<probing_kind == 0,
                             cudax::cuco::linear_probing<cg_size, heterogeneous_hash>,
                             cudax::cuco::double_hashing<cg_size, heterogeneous_hash>>;
  using set_type = cudax::cuco::
    fixed_capacity_set<int, cuda::std::dynamic_extent, cuda::thread_scope_device, heterogeneous_equal, probing_type>;
  constexpr int num_keys = 37;
  test_context context;
  set_type set{context.stream, context.mr, cuda::std::size_t{100}, cudax::cuco::empty_key{-100}, {}, probing_type{}};
  const auto first   = cuda::counting_iterator<int>{0};
  const auto input   = cuda::transform_iterator{first, make_heterogeneous_key<insert_key>{0}};
  const auto queries = cuda::transform_iterator{first, make_heterogeneous_key<query_key>{1}};

  REQUIRE(set.insert(context.stream, input, input + num_keys) == num_keys);

  // Heterogeneous query keys equivalent to stored keys.
  REQUIRE(set.count(context.stream, queries, queries + num_keys) == num_keys);
  REQUIRE(set.count(context.stream, queries + num_keys, queries + 2 * num_keys) == 0);
}
