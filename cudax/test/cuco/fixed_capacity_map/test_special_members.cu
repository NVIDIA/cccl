// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_impl.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cuco = cuda::experimental::cuco;

using map_type  = cuco::fixed_capacity_map<int, int>;
using impl_type = cuco::__open_addressing::__open_addressing_impl<
  map_type::key_type,
  map_type::value_type,
  map_type::thread_scope,
  map_type::key_equal,
  map_type::probing_scheme_type,
  map_type::bucket_size,
  cuda::device_memory_pool_ref>;

C2H_TEST("fixed_capacity_map host ownership and moves", "[container][special_members]")
{
  static_assert(!cuda::std::is_copy_constructible_v<map_type>);
  static_assert(!cuda::std::is_copy_assignable_v<map_type>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<map_type>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<map_type>);

  const cuda::stream stream{cuda::device_ref{0}};
  const auto mr = cuda::device_default_memory_pool(stream.device());
  {
    map_type source{stream, mr, cuda::std::size_t{16}, cuco::empty_key{-1}, cuco::empty_value{-1}};
    const auto first = cuda::constant_iterator{map_type::value_type{0, 7}};
    REQUIRE(source.insert(stream, first, first + 1) == 1);
    const auto* allocation = source.data();
    const auto capacity    = source.capacity();

    map_type moved{cuda::std::move(source)};
    REQUIRE(moved.data() == allocation);
    REQUIRE(moved.capacity() == capacity);

    map_type assigned{stream, mr, cuda::std::size_t{32}, cuco::empty_key{-1}, cuco::empty_value{-1}};
    assigned = cuda::std::move(moved);
    REQUIRE(assigned.data() == allocation);
    REQUIRE(assigned.capacity() == capacity);
    REQUIRE(assigned.insert(stream, first, first + 1) == 0);
  }
  stream.sync();
}

C2H_TEST("fixed_capacity_map implementation moves transfer storage", "[container][special_members]")
{
  static_assert(cuda::std::is_copy_constructible_v<impl_type>);
  static_assert(!cuda::std::is_copy_assignable_v<impl_type>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<impl_type>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<impl_type>);

  const cuda::stream stream{cuda::device_ref{0}};
  const auto mr = cuda::device_default_memory_pool(stream.device());
  {
    impl_type source{
      stream,
      mr,
      cuda::std::size_t{16},
      map_type::value_type{-1, -1},
      map_type::key_equal{},
      map_type::probing_scheme_type{}};
    const auto* allocation = source.data();
    const auto capacity    = source.capacity();

    // A copy fallback would still satisfy is_move_constructible but allocate new storage.
    impl_type moved{cuda::std::move(source)};
    REQUIRE(moved.data() == allocation);
    REQUIRE(moved.capacity() == capacity);

    impl_type assigned{
      stream,
      mr,
      cuda::std::size_t{32},
      map_type::value_type{-1, -1},
      map_type::key_equal{},
      map_type::probing_scheme_type{}};
    assigned = cuda::std::move(moved);
    REQUIRE(assigned.data() == allocation);
    REQUIRE(assigned.capacity() == capacity);
  }
  stream.sync();
}
