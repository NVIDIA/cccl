//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_impl.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

C2H_TEST("fixed_capacity_map dynamic capacity — capacity() reflects the valid capacity", "[capacity][dynamic]")
{
  constexpr ::cuda::std::size_t requested = 1000;
  using dyn_map_t                         = cudax::cuco::fixed_capacity_map<int, int>;
  using impl_type                         = cudax::cuco::__open_addressing::__open_addressing_impl<
    dyn_map_t::key_type,
    dyn_map_t::value_type,
    dyn_map_t::thread_scope,
    dyn_map_t::key_equal,
    dyn_map_t::probing_scheme_type,
    dyn_map_t::bucket_size,
    cuda::device_memory_pool_ref>;

  static_assert(cuda::std::is_copy_constructible_v<impl_type>);
  static_assert(!cuda::std::is_copy_assignable_v<impl_type>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<impl_type>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<impl_type>);

  static_assert(dyn_map_t::capacity_v == ::cuda::std::dynamic_extent,
                "capacity_v must be dynamic_extent for dynamic-capacity maps");
  static_assert(dyn_map_t::ref_type::capacity_v == ::cuda::std::dynamic_extent,
                "ref capacity_v must be dynamic_extent for dynamic maps");

  const auto valid =
    cudax::cuco::make_valid_capacity<dyn_map_t::probing_scheme_type, dyn_map_t::bucket_size>(requested);

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const dyn_map_t map{stream, mr, requested, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
  REQUIRE(map.capacity() >= requested);
}

C2H_TEST("fixed_capacity_map static capacity — valid capacity and capacity_v", "[capacity][static]")
{
  // Double hashing rounds a requested slot count up to a prime-cycle capacity, so the valid capacity
  // must be computed from the probing scheme and bucket size before it can name a static map type.
  using probing                         = cudax::cuco::double_hashing<1, cuda::hash<int>>;
  [[maybe_unused]] constexpr int bucket = 1;

  constexpr ::cuda::std::size_t requested = 1000;
  constexpr auto valid                    = cudax::cuco::make_valid_capacity<probing, bucket>(requested);
  static_assert(valid > requested, "1000 is not a valid double-hashing capacity; it rounds up");

  using smap_t =
    cudax::cuco::fixed_capacity_map<int, int, valid, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing, 1>;
  static_assert(smap_t::capacity_v == valid, "the map type carries the valid capacity, not the request");
  static_assert(smap_t::ref_type::capacity_v == valid, "the ref carries the same valid capacity");

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const smap_t map{stream, mr, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
}

C2H_TEST("fixed_capacity_map dynamic extent — load factor constructor", "[capacity][dynamic][load_factor]")
{
  constexpr int num_elements   = 500;
  constexpr double load_factor = 0.5;

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const cudax::cuco::fixed_capacity_map<int, int> map{
    stream,
    mr,
    static_cast<::cuda::std::size_t>(num_elements),
    load_factor,
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value}};

  // With load_factor = 0.5 and 500 elements, capacity should be >= 1000
  REQUIRE(map.capacity() >= static_cast<::cuda::std::size_t>(num_elements / load_factor));
}

// Offset pool allocations by the requested alignment so the pool's stronger alignment cannot
// conceal an insufficient request. Deallocation must receive the same alignment as allocation.
struct alignment_resource
{
  cuda::device_memory_pool_ref upstream;
  cuda::std::size_t* requested_alignment;

  [[nodiscard]] void* allocate_sync(cuda::std::size_t bytes, cuda::std::size_t alignment)
  {
    *requested_alignment = alignment;
    return static_cast<char*>(upstream.allocate_sync(bytes + alignment, alignment)) + alignment;
  }

  [[nodiscard]] void* allocate(cuda::stream_ref stream, cuda::std::size_t bytes, cuda::std::size_t alignment)
  {
    *requested_alignment = alignment;
    return static_cast<char*>(upstream.allocate(stream, bytes + alignment, alignment)) + alignment;
  }

  void deallocate_sync(void* pointer, cuda::std::size_t bytes, cuda::std::size_t alignment) noexcept
  {
    upstream.deallocate_sync(static_cast<char*>(pointer) - alignment, bytes + alignment, alignment);
  }

  void deallocate(cuda::stream_ref stream, void* pointer, cuda::std::size_t bytes, cuda::std::size_t alignment) noexcept
  {
    upstream.deallocate(stream, static_cast<char*>(pointer) - alignment, bytes + alignment, alignment);
  }

  [[nodiscard]] bool operator==(const alignment_resource& other) const noexcept
  {
    return upstream == other.upstream && requested_alignment == other.requested_alignment;
  }

  [[nodiscard]] bool operator!=(const alignment_resource& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(const alignment_resource&, cuda::mr::device_accessible) noexcept {}
};

template <class Key, class Mapped, cuda::std::size_t Alignment>
struct slot_alignment_case
{
  using key_type                           = Key;
  using mapped_type                        = Mapped;
  static constexpr auto expected_alignment = Alignment;
};

using slot_alignment_cases =
  c2h::type_list<slot_alignment_case<cuda::std::int32_t, cuda::std::int32_t, 8>,
                 slot_alignment_case<cuda::std::uint16_t, cuda::std::uint16_t, 4>,
                 slot_alignment_case<cuda::std::int64_t, cuda::std::int64_t, 8>>;

C2H_TEST("fixed_capacity_map allocation alignment", "[capacity][alignment]", slot_alignment_cases)
{
  using test_case   = c2h::get<0, TestType>;
  using key_type    = typename test_case::key_type;
  using mapped_type = typename test_case::mapped_type;
  using probing     = cudax::cuco::linear_probing<1, cuda::hash<key_type>>;
  using dynamic_map = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    cuda::std::dynamic_extent,
    cuda::thread_scope_device,
    cuda::std::equal_to<key_type>,
    probing,
    1,
    alignment_resource>;
  using static_map = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    17,
    cuda::thread_scope_device,
    cuda::std::equal_to<key_type>,
    probing,
    1,
    alignment_resource>;

  const cuda::stream stream{cuda::device_ref{0}};
  cuda::std::size_t requested_alignment = 0;
  const alignment_resource resource{cuda::device_default_memory_pool(stream.device()), &requested_alignment};
  const auto empty_k = cudax::cuco::empty_key{static_cast<key_type>(-1)};
  const auto empty_v = cudax::cuco::empty_value{mapped_type{}};
  const auto erased  = cudax::cuco::erased_key{static_cast<key_type>(-2)};

  const auto check = [&](auto& map) {
    const auto check_alignment = [&] {
      REQUIRE(requested_alignment == test_case::expected_alignment);
      REQUIRE(reinterpret_cast<cuda::std::uintptr_t>(map.data()) % test_case::expected_alignment == 0);
      REQUIRE(map.capacity() == 17);
    };
    check_alignment();
    using value_type = typename dynamic_map::value_type;
    const auto first = cuda::constant_iterator<value_type>{value_type{key_type{1}, mapped_type{1}}};
    REQUIRE(map.insert(stream, first, first + 1) == 1);
    map.rehash(stream);
    check_alignment();
    REQUIRE(map.insert(stream, first, first + 1) == 0);
  };

  SECTION("dynamic capacity")
  {
    dynamic_map map{stream, resource, 17, empty_k, empty_v};
    check(map);
  }
  SECTION("static capacity")
  {
    static_map map{stream, resource, empty_k, empty_v};
    check(map);
  }
  SECTION("load factor")
  {
    dynamic_map map{stream, resource, 17, 1.0, empty_k, empty_v};
    check(map);
  }
  SECTION("dynamic capacity with erasure")
  {
    dynamic_map map{stream, resource, 17, empty_k, empty_v, erased};
    check(map);
  }
  SECTION("static capacity with erasure")
  {
    static_map map{stream, resource, empty_k, empty_v, erased};
    check(map);
  }
}
