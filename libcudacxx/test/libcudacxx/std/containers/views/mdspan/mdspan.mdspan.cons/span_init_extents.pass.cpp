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

template <class T, class DataHandleT, class SizeType, size_t N, class = void>
struct is_span_cons_avail : cuda::std::false_type
{};

template <class T, class DataHandleT, class SizeType, size_t N>
struct is_span_cons_avail<
  T,
  DataHandleT,
  SizeType,
  N,
  cuda::std::enable_if_t<cuda::std::is_same<
    decltype(T{cuda::std::declval<DataHandleT>(), cuda::std::declval<cuda::std::span<SizeType, N>>()}),
    T>::value>> : cuda::std::true_type
{};

template <class T, class DataHandleT, class SizeType, size_t N>
constexpr bool is_span_cons_avail_v = is_span_cons_avail<T, DataHandleT, SizeType, N>::value;

int main(int, char**)
{
  // extents from cuda::std::span
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = int;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, other_index_t, 2> == true, "");

    cuda::std::array<other_index_t, 1> d{42};
    cuda::std::array<other_index_t, 2> sarr{64, 128};

    mdspan_t m{d.data(), cuda::std::span<other_index_t, 2>{sarr}};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_convertible;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, other_index_t, 2> == false, "");
  }

  // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_convertible;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, other_index_t, 2> == false, "");
  }

  // Constraint: N == rank() || N == rank_dynamic() is true
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = int;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, other_index_t, 1> == false, "");
  }

  // Constraint: is_constructible_v<mapping_type, extents_type> is true
  {
    using mdspan_t = cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_stride>;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, int, 2> == false, "");
  }

  // Constraint: is_default_constructible_v<accessor_type> is true
  {
    using mdspan_t =
      cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_right, Foo::my_accessor<int>>;

    static_assert(is_span_cons_avail_v<mdspan_t, int*, int, 2> == false, "");
  }

  return 0;
}
