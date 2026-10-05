// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__cccl_config>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_set.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <stdexcept>

#include <testing.cuh>

#include "test_common.cuh"

using probing_types =
  c2h::type_list<cudax::cuco::linear_probing<4, cuda::hash<int>>, cudax::cuco::double_hashing<2, cuda::hash<int>>>;

C2H_TEST("fixed_capacity_set capacity constructors", "[container][capacity]", probing_types)
{
  using probing_type                    = c2h::get<0, TestType>;
  constexpr cuda::std::size_t requested = 101;
  constexpr auto valid                  = cudax::cuco::make_valid_capacity<probing_type, 2>(requested);
  using dynamic_set                     = cudax::cuco::fixed_capacity_set<
    int,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuda::std::equal_to<int>,
    probing_type,
    2>;
  using static_set =
    cudax::cuco::fixed_capacity_set<int, valid, cuda::thread_scope_device, cuda::std::equal_to<int>, probing_type, 2>;
  using empty_key       = cudax::cuco::empty_key<int>;
  using stream_ref      = cuda::stream_ref;
  using memory_resource = cuda::device_memory_pool_ref;
  static_assert(dynamic_set::capacity_v == cuda::std::dynamic_extent);
  static_assert(dynamic_set::ref_type::capacity_v == cuda::std::dynamic_extent);
  static_assert(static_set::capacity_v == valid);
  static_assert(static_set::ref_type::capacity_v == valid);
  static_assert(cuda::std::is_constructible_v<dynamic_set, stream_ref, memory_resource, cuda::std::size_t, empty_key>);
  static_assert(!cuda::std::is_constructible_v<dynamic_set, stream_ref, memory_resource, empty_key>);
  static_assert(cuda::std::is_constructible_v<static_set, stream_ref, memory_resource, empty_key>);
  static_assert(!cuda::std::is_constructible_v<static_set, stream_ref, memory_resource, cuda::std::size_t, empty_key>);
  static_assert(!cuda::std::is_copy_constructible_v<dynamic_set>);
  static_assert(!cuda::std::is_copy_assignable_v<dynamic_set>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<dynamic_set>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<dynamic_set>);

  test_context context;
  const dynamic_set dynamic{context.stream, context.mr, requested, empty_key{-1}};
  static_set fixed{context.stream, context.mr, empty_key{-1}};
  REQUIRE(dynamic.capacity() == valid);
  REQUIRE(fixed.capacity() == valid);
  REQUIRE(dynamic.ref().capacity() == dynamic.capacity());
  REQUIRE(fixed.ref().capacity() == fixed.capacity());
  REQUIRE(dynamic.ref().data() == dynamic.data());
  REQUIRE(fixed.ref().data() == fixed.data());

  constexpr int num_keys = 37;
  const auto first       = cuda::counting_iterator<int>{0};
  auto results           = cuda::make_buffer<int>(context.stream, context.mr, num_keys, 0);
  REQUIRE(fixed.insert(context.stream, first, first + num_keys) == num_keys);
  fixed.contains(context.stream, first, first + num_keys, results.begin());
  REQUIRE(context.all_equal(results.data(), num_keys, 1));
  fixed.clear(context.stream);
  fixed.contains(context.stream, first, first + num_keys, results.begin());
  REQUIRE(context.all_equal(results.data(), num_keys, 0));

  const dynamic_set zero{context.stream, context.mr, cuda::std::size_t{0}, empty_key{-1}};
  REQUIRE(zero.capacity() == cudax::cuco::make_valid_capacity<probing_type, 2>(cuda::std::size_t{0}));
  REQUIRE(zero.capacity() > 0);

  constexpr double load_factor = 0.5;
  const dynamic_set scaled{context.stream, context.mr, requested, load_factor, empty_key{-1}};
  REQUIRE(scaled.capacity() == cudax::cuco::make_valid_capacity<probing_type, 2>(requested, load_factor));

#if _CCCL_HAS_EXCEPTIONS()
  constexpr double invalid_factors[] = {0.0, -0.5, 1.1, cuda::std::numeric_limits<double>::quiet_NaN()};
  for (const auto invalid : invalid_factors)
  {
    REQUIRE_THROWS_AS((dynamic_set{context.stream, context.mr, requested, invalid, empty_key{-1}}), std::logic_error);
  }
#endif // _CCCL_HAS_EXCEPTIONS()
}

struct stateful_hash
{
  unsigned seed;

  [[nodiscard]] _CCCL_HOST_DEVICE_API unsigned operator()(int value) const noexcept
  {
    return static_cast<unsigned>(value) ^ seed;
  }
};

struct stateful_equal
{
  int state;

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(int lhs, int rhs) const noexcept
  {
    return lhs == rhs;
  }
};

C2H_TEST("fixed_capacity_set preserves policies and sentinel", "[container][capacity]")
{
  using probing_type = cudax::cuco::linear_probing<1, stateful_hash>;
  using set_type     = cudax::cuco::
    fixed_capacity_set<int, cuda::std::dynamic_extent, cuda::thread_scope_device, stateful_equal, probing_type>;
  test_context context;
  set_type set{
    context.stream,
    context.mr,
    cuda::std::size_t{23},
    cudax::cuco::empty_key{-7},
    stateful_equal{19},
    probing_type{stateful_hash{31}}};
  REQUIRE(set.empty_key_sentinel() == -7);
  REQUIRE(set.key_eq().state == 19);
  REQUIRE(set.hash_function().seed == 31);
  const auto ref = set.ref();
  REQUIRE(ref.empty_key_sentinel() == -7);
  REQUIRE(ref.key_eq().state == 19);
  REQUIRE(ref.hash_function().seed == 31);
  REQUIRE(ref.probing_scheme().hash_function().seed == 31);
  const auto first = cuda::counting_iterator<int>{0};
  REQUIRE(set.insert(context.stream, first, first + 13) == 13);
  auto results = cuda::make_buffer<int>(context.stream, context.mr, 23, 0);
  set.contains(context.stream, first, first + 23, results.begin());
  REQUIRE(context.matches(results.data(), 23, 13));
}

struct allocation_record
{
  cuda::std::size_t bytes     = 0;
  cuda::std::size_t alignment = 0;
};

// Pool allocations are naturally over-aligned, so observe the requested alignment as well.
struct tracking_resource
{
  cuda::device_memory_pool_ref upstream;
  allocation_record* record;

  [[nodiscard]] _CCCL_HOST_API void* allocate_sync(cuda::std::size_t bytes, cuda::std::size_t alignment)
  {
    *record = {bytes, alignment};
    return upstream.allocate_sync(bytes, alignment);
  }

  [[nodiscard]] _CCCL_HOST_API void*
  allocate(cuda::stream_ref stream, cuda::std::size_t bytes, cuda::std::size_t alignment)
  {
    *record = {bytes, alignment};
    return upstream.allocate(stream, bytes, alignment);
  }

  _CCCL_HOST_API void deallocate_sync(void* pointer, cuda::std::size_t bytes, cuda::std::size_t alignment) noexcept
  {
    upstream.deallocate_sync(pointer, bytes, alignment);
  }

  _CCCL_HOST_API void
  deallocate(cuda::stream_ref stream, void* pointer, cuda::std::size_t bytes, cuda::std::size_t alignment) noexcept
  {
    upstream.deallocate(stream, pointer, bytes, alignment);
  }

  [[nodiscard]] _CCCL_HOST_API bool operator==(const tracking_resource& other) const noexcept
  {
    return upstream == other.upstream && record == other.record;
  }

  [[nodiscard]] _CCCL_HOST_API bool operator!=(const tracking_resource& other) const noexcept
  {
    return !(*this == other);
  }

  _CCCL_HOST_API friend void get_property(const tracking_resource&, cuda::mr::device_accessible) noexcept {}
};

template <int Width>
struct weakly_aligned_key
{
  cuda::std::uint8_t bytes[Width];

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator==(const weakly_aligned_key& other) const noexcept
  {
    for (int i = 0; i < Width; ++i)
    {
      if (bytes[i] != other.bytes[i])
      {
        return false;
      }
    }
    return true;
  }
};

static_assert(sizeof(weakly_aligned_key<4>) == 4 && alignof(weakly_aligned_key<4>) == 1);
static_assert(sizeof(weakly_aligned_key<8>) == 8 && alignof(weakly_aligned_key<8>) == 1);
static_assert(cuda::is_bitwise_comparable_v<weakly_aligned_key<4>>);
static_assert(cuda::is_bitwise_comparable_v<weakly_aligned_key<8>>);

template <class Key>
struct make_alignment_key
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API Key operator()(int value) const noexcept
  {
    if constexpr (cuda::std::is_integral_v<Key>)
    {
      return static_cast<Key>(value);
    }
    else
    {
      Key result{};
      for (auto& byte : result.bytes)
      {
        byte = static_cast<cuda::std::uint8_t>(value);
      }
      return result;
    }
  }
};

struct alignment_hash
{
  template <class Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(Key key) const noexcept
  {
    if constexpr (cuda::std::is_integral_v<Key>)
    {
      return static_cast<cuda::std::uint32_t>(key);
    }
    else
    {
      return key.bytes[0];
    }
  }
};

template <class Key>
struct matches_key
{
  Key expected;

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(Key actual) const noexcept
  {
    return actual == expected;
  }
};

template <class Set>
void check_last_slot(Set& set, test_context& context, const allocation_record& record, int capacity)
{
  using key_type           = typename Set::key_type;
  constexpr auto alignment = sizeof(key_type) < 4 ? 4 : sizeof(key_type);
  const auto logical_bytes = static_cast<cuda::std::size_t>(capacity) * sizeof(key_type);
  const auto padded_bytes  = (logical_bytes + alignment - 1) / alignment * alignment;
  REQUIRE(set.capacity() == static_cast<cuda::std::size_t>(capacity));
  REQUIRE(record.alignment >= alignment);
  REQUIRE(record.bytes >= padded_bytes);
  REQUIRE(record.bytes % alignment == 0);
  REQUIRE(reinterpret_cast<cuda::std::uintptr_t>(set.data()) % alignment == 0);

  const auto first =
    cuda::transform_iterator{cuda::counting_iterator<int>{capacity - 1}, make_alignment_key<key_type>{}};
  const auto expected = make_alignment_key<key_type>{}(capacity - 1);
  // Identity hashing places this key directly in the final slot, including capacity one.
  REQUIRE(set.insert(context.stream, first, first + 1) == 1);
  REQUIRE(cuda::std::all_of(
    context.policy(), set.data() + capacity - 1, set.data() + capacity, matches_key<key_type>{expected}));
  auto results = cuda::make_buffer<int>(context.stream, context.mr, 1, 0);
  set.contains(context.stream, first, first + 1, results.begin());
  REQUIRE(context.all_equal(results.data(), 1, 1));
  REQUIRE(set.insert(context.stream, first, first + 1) == 0);
  set.clear_async(context.stream);
  set.insert_async(context.stream, first, first + 1);
  set.contains(context.stream, first, first + 1, results.begin());
  REQUIRE(context.all_equal(results.data(), 1, 1));
}

using alignment_keys =
  c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, weakly_aligned_key<4>, weakly_aligned_key<8>>;
using tail_capacities = c2h::type_list<int_c<1>, int_c<17>>;

C2H_TEST("fixed_capacity_set allocation alignment and final-slot padding",
         "[container][capacity][alignment]",
         alignment_keys,
         tail_capacities)
{
  using key_type         = c2h::get<0, TestType>;
  constexpr int capacity = c2h::get<1, TestType>::value;
  using probing_type     = cudax::cuco::linear_probing<1, alignment_hash>;
  using dynamic_set      = cudax::cuco::fixed_capacity_set<
    key_type,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuda::std::equal_to<key_type>,
    probing_type,
    1,
    tracking_resource>;
  using static_set = cudax::cuco::fixed_capacity_set<
    key_type,
    capacity,
    cuda::thread_scope_device,
    cuda::std::equal_to<key_type>,
    probing_type,
    1,
    tracking_resource>;
  test_context context;
  allocation_record record;
  const tracking_resource resource{context.mr, &record};
  const auto empty = cudax::cuco::empty_key{make_alignment_key<key_type>{}(-1)};

  SECTION("dynamic capacity")
  {
    dynamic_set set{context.stream, resource, cuda::std::size_t{capacity}, empty};
    check_last_slot(set, context, record, capacity);
  }
  SECTION("static capacity")
  {
    static_set set{context.stream, resource, empty};
    check_last_slot(set, context, record, capacity);
  }
  SECTION("load factor")
  {
    dynamic_set set{context.stream, resource, cuda::std::size_t{capacity}, 1.0, empty};
    check_last_slot(set, context, record, capacity);
  }
}
