//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../my_accessor.hpp"
#include "../my_int.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

template <class T, class DataHandleType, class ExtentsType, class = void>
struct is_extents_cons_avail : cuda::std::false_type
{};

template <class T, class DataHandleType, class ExtentsType>
struct is_extents_cons_avail<
  T,
  DataHandleType,
  ExtentsType,
  cuda::std::enable_if_t<
    cuda::std::is_same<decltype(T{cuda::std::declval<DataHandleType>(), cuda::std::declval<ExtentsType>()}), T>::value>>
    : cuda::std::true_type
{};

template <class T, class DataHandleType, class ExtentsType>
constexpr bool is_extents_cons_avail_v = is_extents_cons_avail<T, DataHandleType, ExtentsType>::value;

int main(int, char**)
{
  // extents from extents object
  {
    using ext_t = cuda::std::extents<int, dyn, dyn>;
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan<int, ext_t> m{d.data(), ext_t{64, 128}};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // extents from extents object move
  {
    using ext_t    = cuda::std::extents<int, dyn, dyn>;
    using mdspan_t = cuda::std::mdspan<int, ext_t>;

    static_assert(is_extents_cons_avail_v<mdspan_t, int*, ext_t> == true, "");

    cuda::std::array<int, 1> d{42};
    mdspan_t m{d.data(), cuda::std::move(ext_t{64, 128})};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // Constraint: is_constructible_v<mapping_type, extents_type> is true
  {
    using ext_t    = cuda::std::extents<int, 16, 16>;
    using mdspan_t = cuda::std::mdspan<int, ext_t, cuda::std::layout_stride>;

    static_assert(is_extents_cons_avail_v<mdspan_t, int*, ext_t> == false, "");
  }

  // Constraint: is_default_constructible_v<accessor_type> is true
  {
    using ext_t    = cuda::std::extents<int, 16, 16>;
    using mdspan_t = cuda::std::mdspan<int, ext_t, cuda::std::layout_right, Foo::my_accessor<int>>;

    static_assert(is_extents_cons_avail_v<mdspan_t, int*, ext_t> == false, "");
  }

  return 0;
}
