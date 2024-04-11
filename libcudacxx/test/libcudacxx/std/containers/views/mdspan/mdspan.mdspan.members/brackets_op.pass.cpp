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

#include "../my_int.hpp"

// Will be testing `m[0,0]` when it becomes available
// Alternatively, could use macro `__MDSPAN_OP(m,0,0)` which is turned to either `m[0,0]` or `m(0,0)`,
// depending on if `__cpp_multidimensional_subscript` is defined or not

constexpr auto dyn = cuda::std::dynamic_extent;

template <class, class T, class... OtherIndexTypes>
struct is_bracket_op_avail : cuda::std::false_type
{};

template <class T, class... OtherIndexTypes>
struct is_bracket_op_avail<
  cuda::std::enable_if_t<cuda::std::is_same<decltype(cuda::std::declval<T>()(cuda::std::declval<OtherIndexTypes>()...)),
                                            typename T::accessor_type::reference>::value>,
  T,
  OtherIndexTypes...> : cuda::std::true_type
{};

template <class T, class... OtherIndexTypes>
constexpr bool is_bracket_op_avail_v = is_bracket_op_avail<void, T, OtherIndexTypes...>::value;

template <class T, class OtherIndexType, size_t N, class = void>
struct is_bracket_op_array_avail : cuda::std::false_type
{};

template <class T, class OtherIndexType, size_t N>
struct is_bracket_op_array_avail<
  T,
  OtherIndexType,
  N,
  cuda::std::enable_if_t<
    cuda::std::is_same<decltype(cuda::std::declval<T>()(cuda::std::declval<cuda::std::array<OtherIndexType, N>>())),
                       typename T::accessor_type::reference>::value>> : cuda::std::true_type
{};

template <class T, class OtherIndexType, size_t N>
constexpr bool is_bracket_op_array_avail_v = is_bracket_op_array_avail<T, OtherIndexType, N>::value;

template <class T, class OtherIndexType, size_t N, class = void>
struct is_bracket_op_span_avail : cuda::std::false_type
{};

template <class T, class OtherIndexType, size_t N>
struct is_bracket_op_span_avail<
  T,
  OtherIndexType,
  N,
  cuda::std::enable_if_t<
    cuda::std::is_same<decltype(cuda::std::declval<T>()(cuda::std::declval<cuda::std::span<OtherIndexType, N>>())),
                       typename T::accessor_type::reference>::value>> : cuda::std::true_type
{};

template <class T, class OtherIndexType, size_t N>
constexpr bool is_bracket_op_span_avail_v = is_bracket_op_span_avail<T, OtherIndexType, N>::value;

int main(int, char**)
{
  {
    using element_t = int;
    using index_t   = int;
    using ext_t     = cuda::std::extents<index_t, dyn, dyn>;
    using mdspan_t  = cuda::std::mdspan<element_t, ext_t>;

    cuda::std::array<element_t, 4> d{42, 43, 44, 45};
    mdspan_t m{d.data(), ext_t{2, 2}};

    static_assert(is_bracket_op_avail_v<decltype(m), int, int> == true, "");

    // param pack
    assert(m(0, 0) == 42);
    assert(m(0, 1) == 43);
    assert(m(1, 0) == 44);
    assert(m(1, 1) == 45);

    // array of indices
    assert(m(cuda::std::array<int, 2>{0, 0}) == 42);
    assert(m(cuda::std::array<int, 2>{0, 1}) == 43);
    assert(m(cuda::std::array<int, 2>{1, 0}) == 44);
    assert(m(cuda::std::array<int, 2>{1, 1}) == 45);

    static_assert(is_bracket_op_array_avail_v<decltype(m), int, 2> == true, "");

    // span of indices
    assert(m(cuda::std::span<const int, 2>{cuda::std::array<int, 2>{0, 0}}) == 42);
    assert(m(cuda::std::span<const int, 2>{cuda::std::array<int, 2>{0, 1}}) == 43);
    assert(m(cuda::std::span<const int, 2>{cuda::std::array<int, 2>{1, 0}}) == 44);
    assert(m(cuda::std::span<const int, 2>{cuda::std::array<int, 2>{1, 1}}) == 45);

    static_assert(is_bracket_op_span_avail_v<decltype(m), int, 2> == true, "");
  }

  // Param pack of indices in a type implicitly convertible to index_type
  {
    using element_t = int;
    using index_t   = int;
    using ext_t     = cuda::std::extents<index_t, dyn, dyn>;
    using mdspan_t  = cuda::std::mdspan<element_t, ext_t>;

    cuda::std::array<element_t, 4> d{42, 43, 44, 45};
    mdspan_t m{d.data(), ext_t{2, 2}};

    assert(m(my_int(0), my_int(0)) == 42);
    assert(m(my_int(0), my_int(1)) == 43);
    assert(m(my_int(1), my_int(0)) == 44);
    assert(m(my_int(1), my_int(1)) == 45);
  }

  // Constraint: rank consistency
  {
    using element_t = int;
    using index_t   = int;
    using mdspan_t  = cuda::std::mdspan<element_t, cuda::std::extents<index_t, dyn>>;

    static_assert(is_bracket_op_avail_v<mdspan_t, index_t, index_t> == false, "");

    static_assert(is_bracket_op_array_avail_v<mdspan_t, index_t, 2> == false, "");

    static_assert(is_bracket_op_span_avail_v<mdspan_t, index_t, 2> == false, "");
  }

  // Constraint: convertibility
  {
    using element_t = int;
    using index_t   = int;
    using mdspan_t  = cuda::std::mdspan<element_t, cuda::std::extents<index_t, dyn>>;

    static_assert(is_bracket_op_avail_v<mdspan_t, my_int_non_convertible> == false, "");

    static_assert(is_bracket_op_array_avail_v<mdspan_t, my_int_non_convertible, 1> == false, "");

    static_assert(is_bracket_op_span_avail_v<mdspan_t, my_int_non_convertible, 1> == false, "");
  }

  // Constraint: nonthrow-constructibility
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  {
    using element_t = int;
    using index_t   = int;
    using mdspan_t  = cuda::std::mdspan<element_t, cuda::std::extents<index_t, dyn>>;

    static_assert(is_bracket_op_avail_v<mdspan_t, my_int_non_nothrow_constructible> == false, "");

    static_assert(is_bracket_op_array_avail_v<mdspan_t, my_int_non_nothrow_constructible, 1> == false, "");

    static_assert(is_bracket_op_span_avail_v<mdspan_t, my_int_non_nothrow_constructible, 1> == false, "");
  }
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT

  return 0;
}
