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

template <class, class T, class DataHandleT, class... SizeTypes>
struct is_param_pack_cons_avail : cuda::std::false_type
{};

template <class T, class DataHandleT, class... SizeTypes>
struct is_param_pack_cons_avail<
  cuda::std::enable_if_t<
    cuda::std::is_same<decltype(T{cuda::std::declval<DataHandleT>(), cuda::std::declval<SizeTypes>()...}), T>::value>,
  T,
  DataHandleT,
  SizeTypes...> : cuda::std::true_type
{};

template <class T, class DataHandleT, class... SizeTypes>
constexpr bool is_param_pack_cons_avail_v = is_param_pack_cons_avail<void, T, DataHandleT, SizeTypes...>::value;

int main(int, char**)
{
  {
    using index_t = int;
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan<int, cuda::std::extents<index_t, dyn, dyn>> m{d.data(), 64, 128};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  {
    using index_t = int;
    cuda::std::array<int, 1> d{42};
    cuda::std::
      mdspan<int, cuda::std::extents<index_t, dyn, dyn>, cuda::std::layout_right, cuda::std::default_accessor<int>>
        m{d.data(), 64, 128};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int;

    cuda::std::array<int, 1> d{42};
    mdspan_t m{d.data(), other_index_t(64), other_index_t(128)};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);

    static_assert(is_param_pack_cons_avail_v<mdspan_t, decltype(d.data()), other_index_t, other_index_t> == true, "");
  }

  // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_convertible;

    static_assert(is_param_pack_cons_avail_v<mdspan_t, int*, other_index_t, other_index_t> == false, "");
  }

  // Constraint: (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  {
    using mdspan_t      = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;
    using other_index_t = my_int_non_nothrow_constructible;

    static_assert(is_param_pack_cons_avail_v<mdspan_t, int*, other_index_t, other_index_t> == false, "");
  }
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT

  // Constraint: N == rank() || N == rank_dynamic() is true
  {
    using mdspan_t = cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>>;

    static_assert(is_param_pack_cons_avail_v<mdspan_t, int*, int> == false, "");
  }

  // Constraint: is_constructible_v<mapping_type, extents_type> is true
  {
    using mdspan_t = cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_stride>;

    static_assert(is_param_pack_cons_avail_v<mdspan_t, int*, int> == false, "");
  }

  // Constraint: is_default_constructible_v<accessor_type> is true
  {
    using mdspan_t =
      cuda::std::mdspan<int, cuda::std::extents<int, 16>, cuda::std::layout_right, Foo::my_accessor<int>>;

    static_assert(is_param_pack_cons_avail_v<mdspan_t, int*, int> == false, "");
  }

  return 0;
}
