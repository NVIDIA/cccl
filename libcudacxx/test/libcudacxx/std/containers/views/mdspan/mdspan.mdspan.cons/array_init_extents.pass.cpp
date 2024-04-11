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
struct is_array_cons_avail : cuda::std::false_type
{};

template <class T, class DataHandleT, class SizeType, size_t N>
struct is_array_cons_avail<
  T,
  DataHandleT,
  SizeType,
  N,
  cuda::std::enable_if_t<cuda::std::is_same<
    decltype(T{cuda::std::declval<DataHandleT>(), cuda::std::declval<cuda::std::array<SizeType, N>>()}),
    T>::value>> : cuda::std::true_type
{};

template <class T, class DataHandleT, class SizeType, size_t N>
constexpr bool is_array_cons_avail_v = is_array_cons_avail<T, DataHandleT, SizeType, N>::value;

int main(int, char**)
{
  // extents from cuda::std::array
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>> m{d.data(), cuda::std::array<int, 2>{64, 128}};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // data from cptr, extents from cuda::std::array
  {
    using mdspan_t = cuda::std::mdspan<const int, cuda::std::extents<int, dyn, dyn>>;

    cuda::std::array<int, 1> d{42};
    const int* const ptr = d.data();

    static_assert(is_array_cons_avail_v<mdspan_t, decltype(ptr), int, 2> == true, "");

    mdspan_t m{ptr, cuda::std::array<int, 2>{64, 128}};

    static_assert(cuda::std::is_same<typename decltype(m)::element_type, const int>::value, "");

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_convertible;

    static_assert(is_array_cons_avail_v<mdspan_t, int*, other_index_t, 2> == false, "");
  }

  // Constraint: (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_nothrow_constructible;

    static_assert(is_array_cons_avail_v<mdspan_t, int*, other_index_t, 2> == false, "");
  }
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT

  // Constraint: N == rank() || N == rank_dynamic() is true
  {
    using mdspan_t = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;

    static_assert(is_array_cons_avail_v<mdspan_t, int*, int, 1> == false, "");
  }

  // Constraint: is_constructible_v<mapping_type, extents_type> is true
  {
    using mdspan_t = cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_stride>;

    static_assert(is_array_cons_avail_v<mdspan_t, int*, int, 2> == false, "");
  }

  // Constraint: is_default_constructible_v<accessor_type> is true
  {
    using mdspan_t =
      cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_right, Foo::my_accessor<int>>;

    static_assert(is_array_cons_avail_v<mdspan_t, int*, int, 2> == false, "");
  }

  return 0;
}
